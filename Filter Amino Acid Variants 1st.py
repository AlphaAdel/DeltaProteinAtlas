### Filter Amino Acid Variants (FAAV)
### This file filters the original SNP datasets to get all mutations that have at least one protein-related allele.

import json
import logging
import time
from datetime import datetime
from colorlog import ColoredFormatter
import os
from multiprocessing import Pool, cpu_count, set_start_method

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
logging.info(f'This file filters the original SNP datasets to get all mutations that have at least one protein-related allele.')

# File paths
input_file = '' # Input File Path
output_file = '' # Output File Path

# Start the process
logging.info("Script started.")

start_time = time.time()

# Filter for variants with protein alleles
def filter_aa_variants(lines):
    entries = []
    totalSNPs = 0
    filteredSNPs = 0

    # Load and parse the input JSON data line by line
    for line in lines:
        try:
            entry = json.loads(line) # Load each JSON line as a dictionary
            totalSNPs += 1

            # Check for amino acid mutations in the different alleles
            for mutation_info in entry.get('primary_snapshot_data').get('placements_with_allele'):
                if mutation_info.get('placement_annot').get('mol_type') == "protein":
                    entries.append(entry)
                    filteredSNPs += 1 # Increment the filtered count
                    break

        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON: {e}")

    return entries, totalSNPs, filteredSNPs

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
    first_entry = True

    part_size = 5_000
    batch_size = 32

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        parts = partition(infile, part_size)
        batches = batch_parts(parts, batch_size)

        for batch_num, batch in enumerate(batches, 1):
            with Pool(cpu_count() - 6) as pool:
                results = pool.imap_unordered(filter_aa_variants, batch)
                for part_num, result in enumerate(results, 1):
                    entriesPart, totalSNPsPart, filteredSNPsPart = result
                    totalSNPs += totalSNPsPart
                    filteredSNPs += filteredSNPsPart

                    for entry in entriesPart:
                        if not first_entry:
                            outfile.write("\n")
                        json.dump(entry, outfile)
                        first_entry = False

                    logging.info(f'Batch {batch_num}, Part {part_num} | Filtered Proportion {filteredSNPs} / {totalSNPs} | Filtered Percent {round((filteredSNPs / totalSNPs) * 100, 2)}%')

    # Final logs
    logging.info(f"Filtering completed.")
    logging.info(f"Total number of SNPs: {totalSNPs}.")
    logging.info(f"Number of filtered SNPs: {filteredSNPs}.")

    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Total time taken: {total_time:.2f} seconds.")
