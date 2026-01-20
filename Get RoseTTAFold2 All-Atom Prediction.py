### Get RoseTTAFold2 All-Atom (RFAA) Predictions
### This file uses the PDB protein sequences that were previously curated and extracts RFAA's structure predictions for each of them.
### These predictions will be used with AlphaFold and ESM3 predictions for the DeltaProtein model.

import logging
import subprocess
import sys
import time
from datetime import datetime
from colorlog import ColoredFormatter
import os
from multiprocessing import Pool, set_start_method
import tempfile
import shutil
import psutil

INPUT_FASTA = "/data/PDBTrainablePrtnSeqs_Part3RFAA.fasta"
OUTPUT_ROOT = "/data/RFAAOutputs_Part3"
RFAA_ROOT = "/data/work/rfaa"
LOG_ROOT = "/data/work/logs"
PDB100_BASE = "/data/db/pdb100_2021Mar03/pdb100_2021Mar03/pdb100_2021Mar03"
CHECKPOINT_PATH = "/data/work/rfaa/checkpoints/RFAA_paper_weights.pt"

PART_LINES = 5_000
BATCH_PARTS = 16
JOBS_PER_GPU = 8
GPU_IDS = os.environ.get("RFAA_GPU_IDS", "0,1")
TIMEOUT_SEC = 1_800

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

def verifyEnvironment():
    verified = True
    def pathExists(p):
        return os.path.exists(p)
    def findExecPath(x):
        return shutil.which(x) is not None

    tools = ["mmseqs", "colabfold_search", "hhsearch"]
    missing = [t for t in tools if not findExecPath(t)]
    if missing:
        logging.error(f"ERROR - Missing tools: {missing}. Please ensure they are installed and activated.")
        verified = False
    else:
        logging.info("All tools installed and activated.")

    ffindex = PDB100_BASE + "_pdb.ffindex"
    if not pathExists(ffindex):
        logging.error(f"ERROR - PDB100 ffindex file not found: {ffindex}")
        verified = False
    else:
        logging.info(f"PDB100 verified: {ffindex}")

    if not pathExists(CHECKPOINT_PATH):
        logging.error(f"ERROR - RFAA weights checkpoint not found: {CHECKPOINT_PATH}")
        verified = False
    else:
        logging.info(f"Weights checkpoint verified: {CHECKPOINT_PATH}")

    return verified

def formatWithGB(numBytes):
    return f"{numBytes/1024/1024/1024:.2f} GB"

def getGPUStats():
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits"
        ], text=True)
    except Exception:
        return []

    stats = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        idx, _, gpuUtil, _, memUsed, memTotal, temp = parts
        try:
            stats.append({
                "index": int(idx),
                "gpuUtil": int(gpuUtil),
                "memUsedGB": float(memUsed) / 1024.0,
                "memTotalGB": float(memTotal) / 1024.0,
                "temp": int(temp),
            })
        except ValueError:
            continue

    return stats

def systemStatsLog():
    cpuUsage = psutil.cpu_percent(interval=0.2)
    memUsage = psutil.virtual_memory()
    ramUsage = f"{memUsage.percent}% ({formatWithGB(memUsage.used)}/{formatWithGB(memUsage.total)})"

    try:
        diskUsage = shutil.disk_usage("/data")
        usedGB = diskUsage.total - diskUsage.free
        diskPct = int(usedGB * 100 / diskUsage.total) if diskUsage.total else 0
        disk = f"{diskPct}% ({formatWithGB(usedGB)}/{formatWithGB(diskUsage.total)})"
    except Exception:
        disk = 'n/a'

    gpuStats = getGPUStats()
    if gpuStats:
        gparts = []
        for g in gpuStats:
            gparts.append(f"GPU{g['index']} {g['gpuUtil']}% | {g['memUsedGB']:.1f}/{g['memTotalGB']:.1f} GiB | {g['temp']}°C")
        gpu = ", ".join(gparts)
    else:
        gpu = "n/a"

    return f"CPU Usage: {cpuUsage}% | RAM Usage: {ramUsage} | Disk Space: {disk} | GPU Stats: {gpu}"

def runRFAA(task):
    seqID, seq, gpuID, cpusPerJob = task
    outPDB = os.path.join(OUTPUT_ROOT, f"ID_{seqID}.pdb")
    if os.path.exists(outPDB):
        return True

    tempFastaFile = tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False)
    tempFastaPath = tempFastaFile.name
    tempFastaFile.write(f">{seqID}\n{seq}\n")
    tempFastaFile.close()

    cmd = [
        sys.executable, "-m", "rf2aa.run_inference",
        "--config-name", "protein",
        f"protein_inputs.A.fasta_file={tempFastaPath}",
        f"output_path={OUTPUT_ROOT}",
        f"job_name=ID_{seqID}",

        f"database_params.hhdb={PDB100_BASE}",
        f"++checkpoint_path={CHECKPOINT_PATH}",
        f"database_params.num_cpus={cpusPerJob}",
        f"database_params.command=make_msa_runner.sh",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpuID)
    env["OMP_NUM_THREADS"] = str(cpusPerJob)
    env["MKL_NUM_THREADS"] = str(cpusPerJob)
    env["OPENBLAS_NUM_THREADS"] = str(cpusPerJob)
    env["PYTHONWARNINGS"] = "ignore:::hydra._internal.defaults_list"

    try:
        subprocess.run(cmd, env=env, cwd=RFAA_ROOT, check=True, timeout=TIMEOUT_SEC)
    except subprocess.CalledProcessError as e:
        logging.error(f"ERROR - RFAA failed to predict structure for {seqID} due to {e}.")
        return False
    except Exception as e:
        logging.error(f"ERROR - RFAA failed to predict structure for {seqID} due to {e}.")
        return False
    finally:
        try:
            os.unlink(tempFastaPath)
        except FileNotFoundError:
            pass

    if os.path.exists(outPDB):
        return True
    else:
        logging.error(f"ERROR - Sequence with PDB ID {seqID} failed to be processed by RFAA.")
        return False

def streamFastaRecords(fastaPath):
    currID, currSeq = None, []

    def flush():
        nonlocal currID, currSeq
        if currID and currSeq:
            yield currID, "".join(currSeq)
        currID, currSeq = None, []

    with open(fastaPath, "r") as fh:
        for batch in batch_parts(partition(fh, PART_LINES), BATCH_PARTS):
            for line in (ln for part in batch for ln in part):
                lineStrip = line.strip()
                if not lineStrip:
                    continue
                if lineStrip.startswith(">"):
                    for rec in flush():
                        yield rec
                    currID = lineStrip[1:].split()[0].replace("/", "_").replace("\\", "_").replace(":", "_")[:200]
                    currSeq = []
                else:
                    currSeq.append(lineStrip)

    yield from flush()

if __name__ == "__main__":
    # Color logging
    LOG_LEVEL = logging.INFO
    LOGFORMAT = "%(log_color)s%(asctime)s - %(levelname)s - %(message)s"

    formatter = ColoredFormatter(
        LOGFORMAT,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red",
        }
    )
    color_handler = logging.StreamHandler()
    color_handler.setFormatter(formatter)

    os.makedirs(LOG_ROOT, exist_ok=True)
    file_handler = logging.FileHandler(f"{LOG_ROOT}/{os.path.basename(__file__)} {datetime.now().date()} {datetime.now().strftime('%H.%M.%S')}.txt")
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOGFORMAT,
        handlers=[file_handler, color_handler]
    )
    logging.info(f'{os.path.basename(__file__)}')
    logging.info(f'{datetime.now()}')
    logging.info(f'This file uses the PDB protein sequences that were previously curated and extracts RFAA\'s structure predictions for each of them. These predictions will be used with AlphaFold and ESM3 predictions for the DeltaProtein model.')

    logging.info("Script started.")
    start_time = time.time()

    try:
        set_start_method("spawn")
    except RuntimeError:
        pass

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    gpuIDs = [g.strip() for g in GPU_IDS.split(",") if g.strip()]
    numWorkers = max(1, len(gpuIDs) * JOBS_PER_GPU)
    SUBMIT = max(4, 4 * numWorkers)
    totalCPUs = os.cpu_count() or 28
    cpusPerJob = max(4, totalCPUs // numWorkers)
    logging.info(f"Number of GPUs: {gpuIDs} | Number of workers: {numWorkers} | Number of CPUs per job: {cpusPerJob}")
    logging.info("Initial system usage: " + systemStatsLog())

    if not verifyEnvironment():
        logging.error("ERROR - Environment has issues, aborting process.")
        sys.exit(1)

    successful = 0
    unsuccessful = 0
    gpuIndex = 0
    tasks = []

    with Pool(processes=numWorkers) as pool:
        for seqID, seq in streamFastaRecords(INPUT_FASTA):
            outPDB = os.path.join(OUTPUT_ROOT, f"ID_{seqID}.pdb")
            if os.path.exists(outPDB):
                successful += 1
                continue

            gpuID = gpuIDs[gpuIndex % len(gpuIDs)]
            gpuIndex += 1

            tasks.append((seqID, seq, gpuID, cpusPerJob))
            if len(tasks) >= SUBMIT:
                for ok in pool.imap_unordered(runRFAA, tasks, chunksize=1):
                    successful += 1 if ok else 0
                    unsuccessful += 0 if ok else 1
                tasks.clear()
                logging.info(f"Successfully processed {successful} sequences | Unsuccessfully processed {unsuccessful} sequences | Usage: {systemStatsLog()}")

        if tasks:
            for ok in pool.map(runRFAA, tasks):
                successful += 1 if ok else 0
                unsuccessful += 0 if ok else 1
            logging.info(f"Successfully processed {successful} sequences | Unsuccessfully processed {unsuccessful} sequences | Usage: {systemStatsLog()}")

    logging.info("RFAA predictions completed.")
    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Total number of successfully processed protein sequences: {successful}")
    logging.info(f"Total number of unsuccessfully processed protein sequences: {unsuccessful}")
    logging.info(f"Final system usage: {systemStatsLog()}")
    logging.info(f"Total time taken: {total_time:.2f} seconds.")
