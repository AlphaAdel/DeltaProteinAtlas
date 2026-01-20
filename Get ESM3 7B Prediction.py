### Get ESM3 7B Predictions
### This file uses the PDB protein sequences that were previously curated and extracts ESM3-7B's structure predictions for each of them.
### These predictions will be used with AlphaFold and RoseTTAFold predictions for the DeltaProtein model.

import logging
import time
from datetime import datetime, timedelta, timezone
from colorlog import ColoredFormatter
import os
from multiprocessing import Pool, cpu_count, set_start_method
from esm.sdk import client
from esm.sdk.api import ESMProtein, GenerationConfig
import torch
from warnings import filterwarnings

filterwarnings("ignore", message=r".*Entity ID not found in metadata.*", category=UserWarning)
inputFile = '' # Input File Path
outputDirectory = '' # Output File Path

# Code parameters
MODEL = None
NUM_STEPS = 8
SUBMIT = 32
LINES_PER_PART = 5_000
PARTS_PER_BATCH = 16
DAILY_LIMIT = 10_000 * 10_000
RPM_LIMIT = 50

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
    global MODEL
    if MODEL is None:
        apiKey = os.environ.get("ESM_API_KEY")
        MODEL = client(model="esm3-medium-2024-08", token=apiKey)

    seqID, seq = task
    outPath = os.path.join(outputDirectory, f"{seqID}.pdb")
    if os.path.exists(outPath):
        return True

    clean = sanitize(seq)
    if not clean:
        return False

    try:
        proteinSeq = ESMProtein(sequence=clean)
        proteinStruct = MODEL.generate(proteinSeq, GenerationConfig(track="structure", num_steps=NUM_STEPS))

        if hasattr(proteinStruct, "to_pdb"):
            proteinStruct.to_pdb(outPath)
            return True

        err_msg = getattr(proteinStruct, "error_msg", None)
        detail = getattr(proteinStruct, "detail", None) or getattr(proteinStruct, "message", None) or getattr(proteinStruct, "error", None)
        logging.error(f"ERROR - Failed to get ESM3 predictions for sequence {seqID} due to {err_msg or detail}")
        return False
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

def secsTilMidnight():
    now = datetime.now(timezone.utc)
    nxt = datetime.combine((now + timedelta(days=1)).date(), datetime.min.time(), tzinfo=timezone.utc)
    return max(0, int((nxt - now).total_seconds()))

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
    logging.info(f'This file uses the PDB protein sequences that were previously curated and extracts ESM3-7B\'s structure predictions for each of them. These predictions will be used with AlphaFold and RoseTTAFold predictions for the DeltaProtein model.')

    logging.info("Script started.")
    start_time = time.time()

    # Increase the number of threads per process rather than more processes with fewer threads for better resource management
    os.environ["OMP_NUM_THREADS"] = f"{cpu_count() // 2}"
    os.environ["MKL_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    torch.set_num_interop_threads(1)

    try:
        set_start_method('spawn')
    except RuntimeError:
        logging.error("ERROR - Start method has already been set.")

    successful = 0
    unsuccessful = 0
    tokensSubmitted = 0

    requestsMade = 0
    requestStartTime = time.time()
    with Pool(processes=3) as pool:
        tasks = []
        for seqID, seq in streamFastaRecords(inputFile):
            if os.path.exists(os.path.join(outputDirectory, f"{seqID}.pdb")):
                successful += 1
                continue

            cleanSeq = sanitize(seq)
            if not cleanSeq:
                continue

            if tokensSubmitted + len(cleanSeq) > DAILY_LIMIT:
                if tasks:
                    currentTime = time.time()
                    elapsed = currentTime - requestStartTime
                    if elapsed >= 60:
                        requestStartTime = currentTime
                        requestsMade = 0
                    taskLen = len(tasks)
                    if requestsMade + taskLen > RPM_LIMIT:
                        waitTime = max(0.0, 60 - elapsed)
                        if waitTime > 0:
                            time.sleep(waitTime)
                        requestStartTime = time.time()
                        requestsMade = 0

                    for success in pool.map(getESM3Predictions, tasks):
                        successful += 1 if success else 0
                        unsuccessful += 0 if success else 1
                    requestsMade += taskLen
                    tasks.clear()
                    logging.info(f"Successfully processed {successful} sequences | Unsuccessfully processed {unsuccessful} sequences")

                sleepTime = secsTilMidnight()
                logging.info(f"Daily token limit reached. Sleeping until reset in {int(sleepTime // 3600)}:{int((sleepTime % 3600) // 60)}:{int(sleepTime % 60)}...")
                time.sleep(sleepTime + 10)

                tokensSubmitted = 0
                requestsMade = 0
                requestStartTime = time.time()

            tokensSubmitted += len(cleanSeq)
            tasks.append((seqID, seq))

            if len(tasks) >= SUBMIT:
                currentTime = time.time()
                elapsed = currentTime - requestStartTime
                if elapsed >= 60:
                    requestStartTime = currentTime
                    requestsMade = 0
                taskLen = len(tasks)
                if requestsMade + taskLen > RPM_LIMIT:
                    waitTime = max(0.0, 60 - elapsed)
                    if waitTime > 0:
                        time.sleep(waitTime)
                    requestStartTime = time.time()
                    requestsMade = 0

                for success in pool.map(getESM3Predictions, tasks):
                    successful += 1 if success else 0
                    unsuccessful += 0 if success else 1
                requestsMade += taskLen
                tasks.clear()
                logging.info(f"Successfully processed {successful} sequences | Unsuccessfully processed {unsuccessful} sequences")

        if tasks:
            currentTime = time.time()
            elapsed = currentTime - requestStartTime
            if elapsed >= 60:
                requestStartTime = currentTime
                requestsMade = 0
            taskLen = len(tasks)
            if requestsMade + taskLen > RPM_LIMIT:
                waitTime = max(0.0, 60 - elapsed)
                if waitTime > 0:
                    time.sleep(waitTime)
                requestStartTime = time.time()
                requestsMade = 0

            for success in pool.map(getESM3Predictions, tasks):
                successful += 1 if success else 0
                unsuccessful += 0 if success else 1
            requestsMade += taskLen
            logging.info(f"Successfully processed {successful} sequences | Unsuccessfully processed {unsuccessful} sequences")

    # Final logs
    logging.info("ESM3 predictions completed.")
    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Total number of successfully processed protein sequences: {successful}")
    logging.info(f"Total number of unsuccessfully processed protein sequences: {unsuccessful}")
    logging.info(f"Total time taken: {total_time:.2f} seconds.")
