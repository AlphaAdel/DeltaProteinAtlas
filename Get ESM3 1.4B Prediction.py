### Get ESM3 1.4B Predictions
### This file uses the PDB protein sequences that were previously curated and extracts ESM3's structure predictions for each of them.
### These predictions will be used with AlphaFold and RoseTTAFold predictions for the DeltaProtein model.
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import gc
import logging
import time
from datetime import datetime
from colorlog import ColoredFormatter
from multiprocessing import set_start_method, get_context
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig
from huggingface_hub import login
import torch
from warnings import filterwarnings

filterwarnings("ignore", message=r".*Entity ID not found in metadata.*", category=UserWarning)
inputFile = '' # Input File Path
outputDirectory = '' # Output File Path

# Code parameters
MODEL = None
DEVICE = "cpu"
NUM_STEPS = 4
LINES_PER_PART = 1_000
PARTS_PER_BATCH = 8
STARTUP_SEM = None
WARMED_UP = False
LEN_LIMIT = 4096
LOG_EVERY = 32

def initWorker(sem):
    global MODEL, DEVICE, STARTUP_SEM, WARMED_UP
    STARTUP_SEM = sem
    WARMED_UP = False

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    STARTUP_SEM.acquire()
    try:
        MODEL = ESM3.from_pretrained("esm3-open").to(DEVICE)
        MODEL.eval()
    finally:
        STARTUP_SEM.release()

def partition(file, size):
    part = []
    for line in file:
        part.append(line)
        if len(part) == size:
            yield part
            part = []
    if part:
        yield part

def batch_parts(parts, batch_size):
    batch = []
    for item in parts:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def sanitize(seq):
    seq = seq.upper()
    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    cleanSeq = "".join(char for char in seq if char in allowed)
    return cleanSeq if cleanSeq else None

def getESM3Predictions(task):
    filterwarnings("ignore", message=r".*Entity ID not found in metadata.*", category=UserWarning)

    seqID, seq = task
    outPath = os.path.join(outputDirectory, f"{seqID}.pdb")
    if os.path.exists(outPath):
        return True

    clean = sanitize(seq)
    if not clean:
        return False

    try:
        logging.info(f"Generating structure predictions for sequence ID {seqID}...")
        global STARTUP_SEM, WARMED_UP

        if not WARMED_UP:
            STARTUP_SEM.acquire()
            try:
                if not WARMED_UP:
                    with torch.inference_mode():
                        proteinSeq = ESMProtein(sequence=clean)
                        proteinStruct = MODEL.generate(proteinSeq, GenerationConfig(track="structure", num_steps=NUM_STEPS))
                        proteinStruct.to_pdb(outPath)
                    del proteinSeq, proteinStruct
                    if (hash(seqID) & 0x3F) == 0:
                        gc.collect()
                    logging.info(f"Successfully generated structure predictions for sequence ID {seqID}")
                    WARMED_UP = True
                    return True
            finally:
                STARTUP_SEM.release()

        with torch.inference_mode():
            proteinSeq = ESMProtein(sequence=clean)
            proteinStruct = MODEL.generate(proteinSeq, GenerationConfig(track="structure", num_steps=NUM_STEPS))
            proteinStruct.to_pdb(outPath)
        del proteinSeq, proteinStruct
        if (hash(seqID) & 0x3F) == 0:
            gc.collect()
        logging.info(f"Successfully generated structure predictions for sequence ID {seqID}")
        return True
    except Exception as e:
        logging.error(f"ERROR - Failed to get ESM3 predictions for sequence {seqID} due to {e}")
        return False

def streamFastaRecords(fastaPath):
    currID, currSeq = None, []

    def yieldRecords():
        nonlocal currID, currSeq
        if currID and currSeq:
            yield currID, "".join(currSeq)
        currID, currSeq = None, []

    with open(fastaPath, "r") as file:
        for batch in batch_parts(partition(file, LINES_PER_PART), PARTS_PER_BATCH):
            for line in (ln for part in batch for ln in part):
                lineStrip = line.strip()
                if not lineStrip:
                    continue
                if lineStrip.startswith(">"):
                    for record in yieldRecords():
                        yield record

                    currID = lineStrip[1:].split()[0].replace("/", "_").replace("\\", "_").replace(":", "_")[:200]
                    currSeq = []
                else:
                    currSeq.append(lineStrip)

    for record in yieldRecords():
        yield record

if __name__ == "__main__":
    # Color logging
    LOG_LEVEL = logging.INFO
    LOGFORMAT = "%(log_color)s%(asctime)s - %(levelname)s - %(message)s"

    formatter = ColoredFormatter(
        LOGFORMAT,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "white",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        }
    )
    color_handler = logging.StreamHandler()
    color_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(f'Project Delta Logs/{os.path.basename(__file__)} {datetime.now().date()} {datetime.now().strftime('%H.%M.%S')}.txt')
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOGFORMAT,
        handlers=[file_handler, color_handler]
    )
    logging.info(f'{os.path.basename(__file__)}')
    logging.info(f'{datetime.now()}')
    logging.info(f'This file uses the PDB protein sequences that were previously curated and extracts ESM3\'s structure predictions for each of them. These predictions will be used with AlphaFold and RoseTTAFold predictions for the DeltaProtein model.')

    logging.info("Script started.")
    start_time = time.time()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    try:
        set_start_method('spawn')
    except RuntimeError:
        logging.error("ERROR - Start method has already been set.")

    login(token='###', add_to_git_credential=False)
    context = get_context("spawn")
    sem = context.Semaphore(1)

    successful = 0
    unsuccessful = 0
    with context.Pool(processes=2, initializer=initWorker, initargs=(sem,), maxtasksperchild=100) as pool:
        def taskLoader():
            global successful, unsuccessful
            for seqID, seq in streamFastaRecords(inputFile):
                if not os.path.exists(os.path.join(outputDirectory, f"{seqID}.pdb")):
                    cleanSeq = sanitize(seq)
                    seqLen = len(cleanSeq) if cleanSeq else 0
                    if seqLen >= LEN_LIMIT:
                        unsuccessful += 1
                        continue

                    yield seqID, seq
                else:
                    successful += 1
                    logging.info(f"Successfully processed {successful} sequences | Unsuccessfully processed {unsuccessful} sequences")

        for success in pool.imap_unordered(getESM3Predictions, taskLoader(), chunksize=1):
            successful += 1 if success else 0
            unsuccessful += 0 if success else 1
            if (successful + unsuccessful) % LOG_EVERY == 0:
                logging.info(f"Successfully processed {successful} sequences | Unsuccessfully processed {unsuccessful} sequences")

    # Final logs
    logging.info("ESM3 predictions completed.")
    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Total number of successfully processed protein sequences: {successful}")
    logging.info(f"Total number of unsuccessfully processed protein sequences: {unsuccessful}")
    logging.info(f"Total time taken: {total_time:.2f} seconds.")
