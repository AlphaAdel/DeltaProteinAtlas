### Filter Amino Acid Variants 4 (FAAV4)
### This file filters the FAAV3 SNP datasets to get all mutations that have at least one protein-related association with a clinical significance of pathogenic, pathogenic-likely-pathogenic, or likely-pathogenic.

import ujson as json
import logging
import time
from datetime import datetime
from colorlog import ColoredFormatter
import os
from multiprocessing import Pool, cpu_count, set_start_method
from collections import defaultdict, Counter

# Color logging
LOG_LEVEL = logging.INFO
LOGFORMAT = "%(log_color)s%(asctime)s - %(levelname)s - %(message)s"

formatter = ColoredFormatter(
    LOGFORMAT,
    log_colors={
        "DEBUG":    "cyan",
        "INFO":     "white",
        "WARNING":  "yellow",
        "ERROR":    "red",
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
logging.info(f'This file filters the FAAV3 SNP datasets to get all mutations that have at least one protein-related association with a clinical significance of pathogenic, pathogenic-likely-pathogenic, or likely-pathogenic.')

# File paths
input_file = '' # Input File Path
output_file = '' # Output File Path

# Start the process
logging.info("Script started.")

start_time = time.time()

def filter_aa_variants(lines):
    entries = []
    diseaseSig = defaultdict(int)
    totalSNPs = 0
    filteredSNPs = 0

    for line in lines:
        found = False
        entry = json.loads(line)
        totalSNPs += 1

        for alleleAnnot in entry.get('PrimarySnapshotData').get('AlleleAnnotations'):
            if any(alleleAnnot.get('clinical')):
                for assemblyAnnot in alleleAnnot.get('assembly_annotation'):
                    for gene in assemblyAnnot.get('genes'):
                        for proteinMutation in gene.get('rnas'):
                            if proteinMutation.get('protein', {}):
                                for diseaseEntry in alleleAnnot.get('clinical'):
                                    for sigType in diseaseEntry.get('clinical_significances'):
                                        if sigType == 'pathogenic' or sigType == 'pathogenic-likely-pathogenic' or sigType == 'likely-pathogenic':
                                            entries.append(entry)
                                            filteredSNPs += 1
                                            diseaseSig[sigType] += 1
                                            found = True
                                            break
                                    if found:
                                        break
                                if found:
                                    break
                        if found:
                            break
                    if found:
                        break
            if found:
                break

    return entries, totalSNPs, filteredSNPs, dict(diseaseSig)

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

if __name__ == "__main__":
    set_start_method('fork')

    totalSNPs = 0
    filteredSNPs = 0
    diseaseSig = {}
    first_entry = True

    part_size = 1_000
    batch_size = 16

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        parts = partition(infile, part_size)
        batches = batch_parts(parts, batch_size)

        with Pool(cpu_count() - 6) as pool:
            for batch_num, batch in enumerate(batches, 1):
                results = pool.imap_unordered(filter_aa_variants, batch)
                for part_num, result in enumerate(results, 1):
                    entriesPart, totalSNPsPart, filteredSNPsPart, diseaseSigPart = result
                    totalSNPs += totalSNPsPart
                    filteredSNPs += filteredSNPsPart
                    diseaseSig = dict(Counter(diseaseSig) + Counter(diseaseSigPart))

                    for entry in entriesPart:
                        if not first_entry:
                            outfile.write("\n")
                        json.dump(entry, outfile)
                        first_entry = False

                    logging.info(f'Batch {batch_num}, Part {part_num} | Filtered {totalSNPs} Variants')

    # Final logs
    logging.info(f"Filtering completed.")
    logging.info(f"Number of SNPs per disease significance type: \n{'\n'.join([f'{key}: {value}' for key, value in diseaseSig.items()])}")
    logging.info(f"Total number of SNPs: {totalSNPs}.")
    logging.info(f"Total number of filtered SNPs: {filteredSNPs}")
    logging.info(f"Total number of disease significance types logged: {sum(diseaseSig.values())}")

    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Total time taken: {total_time:.2f} seconds.")
