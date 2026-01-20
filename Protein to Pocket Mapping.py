### Protein to Pocket Mapping
### This file reads per-protein Fpocket outputs and produces a single JSON file containing every protein, every pocket, pocket statistics, amino-acid ranges, and the pocket structure path.

import os
import time
import json
import re
import logging
from datetime import datetime
from colorlog import ColoredFormatter
from concurrent.futures import ThreadPoolExecutor, as_completed

inputDirectory = '' # Input Path
outputFile = '' # Output Path
concurrentFolderParses = 14

pocketHeaderRegex = re.compile(r'^\s*Pocket\s+(\d+)\s*:\s*$')
keyValueRegex = re.compile(r'^\s*([^:]+?)\s*:\s*(.*?)\s*$')
infoFileRegex = re.compile(r'.+_info\.txt$', re.IGNORECASE)

keyMap = {
    'Score': 'score',
    'Druggability Score': 'druggabilityScore',
    'Number of Alpha Spheres': 'numAlphaSpheres',
    'Total SASA': 'totalSasa',
    'Polar SASA': 'polarSasa',
    'Apolar SASA': 'apolarSasa',
    'Volume': 'volume',
    'Mean local hydrophobic density': 'meanLocalHydrophobicDensity',
    'Mean alpha sphere radius': 'meanAlphaSphereRadius',
    'Mean alp. sph. solvent access': 'meanAlphaSphereSolventAccess',
    'Apolar alpha sphere proportion': 'apolarAlphaSphereProportion',
    'Hydrophobicity score': 'hydrophobicityScore',
    'Volume score': 'volumeScore',
    'Polarity score': 'polarityScore',
    'Charge score': 'chargeScore',
    'Proportion of polar atoms': 'proportionPolarAtoms',
    'Alpha sphere density': 'alphaSphereDensity',
    'Cent. of mass - Alpha Sphere max dist': 'centerOfMassToAlphaSphereMaxDist',
    'Flexibility': 'flexibility',
}

def cleanLabel(label):
    return ' '.join(label.strip().split()).rstrip(':')

def normalizeKey(label):
    label = cleanLabel(label)
    if label in keyMap:
        return keyMap[label]

    parts = re.split(r'[^A-Za-z0-9]+', label)
    parts = [p for p in parts if p]
    if not parts:
        return 'unknown'
    return parts[0].lower() + ''.join(p[:1].upper() + p[1:] for p in parts[1:])

def parseNumber(valueText):
    valueText = valueText.strip().replace('\t', ' ')
    if valueText == '':
        return ''

    upper = valueText.upper()
    if upper in {'NA', 'N/A', 'NONE'}:
        return valueText

    if re.fullmatch(r'[+-]?\d+', valueText):
        try:
            return int(valueText)
        except Exception:
            return valueText

    if re.fullmatch(r'[+-]?\d+(\.\d+)?([eE][+-]?\d+)?', valueText):
        try:
            return float(valueText)
        except Exception:
            return valueText

    return valueText

def parseFpocketInfoFile(infoFilePath):
    pocketStatsById = {}
    currentPocketId = None

    with open(infoFilePath, 'r', errors='replace') as f:
        for rawLine in f:
            line = rawLine.rstrip('\n')

            headerMatch = pocketHeaderRegex.match(line)
            if headerMatch:
                currentPocketId = int(headerMatch.group(1))
                pocketStatsById.setdefault(currentPocketId, {})
                continue

            if currentPocketId is None:
                continue

            kvMatch = keyValueRegex.match(line)
            if not kvMatch:
                continue

            rawLabel = kvMatch.group(1)
            rawValue = kvMatch.group(2)

            label = cleanLabel(rawLabel)
            key = normalizeKey(label)
            value = parseNumber(rawValue)

            pocketStatsById[currentPocketId][key] = value

    return pocketStatsById

def parsePocketAtmResidues(pocketAtmPdbPath):
    residues = set()

    with open(pocketAtmPdbPath, 'r', errors='replace') as f:
        for line in f:
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                continue

            resName = line[17:20].strip()
            resSeqText = line[22:26].strip()
            iCode = line[26].strip()

            if resSeqText == '':
                continue

            try:
                resSeq = int(resSeqText)
            except ValueError:
                continue

            residues.add((resSeq, iCode, resName))

    sortedResidues = sorted(residues, key=lambda x: (x[0], x[1], x[2]))
    return sortedResidues

def toContiguousRanges(sortedResidueNumbers):
    if not sortedResidueNumbers:
        return []

    ranges = []
    start = sortedResidueNumbers[0]
    prev = sortedResidueNumbers[0]
    for value in sortedResidueNumbers[1:]:
        if value <= (prev + 1):
            prev = value
        else:
            ranges.append((start, prev))
            start = value
            prev = value

    ranges.append((start, prev))
    return ranges

def buildAminoAcidRangesFromPocketAtm(pocketAtmPdbPath):
    residues = parsePocketAtmResidues(pocketAtmPdbPath)
    ranges = []
    residueNumbers = sorted({resSeq for (resSeq, iCode, resName) in residues})
    for (start, end) in toContiguousRanges(residueNumbers):
        ranges.append({
            'start': start,
            'end': end
        })

    return ranges

def discoverProteinOutputFolders():
    proteinFolders = []
    if not os.path.isdir(inputDirectory):
        logging.error(f'Input directory does not exist: {inputDirectory}')
        return proteinFolders
    for entry in sorted(os.listdir(inputDirectory)):
        fullPath = os.path.join(inputDirectory, entry)
        if not os.path.isdir(fullPath):
            continue
        proteinFolders.append(fullPath)

    return proteinFolders

def findInfoFileInProteinFolder(proteinFolderPath, proteinId):
    preferred = os.path.join(proteinFolderPath, f'{proteinId}_info.txt')
    if os.path.isfile(preferred):
        return preferred

    for entry in os.listdir(proteinFolderPath):
        fullPath = os.path.join(proteinFolderPath, entry)
        if os.path.isfile(fullPath) and infoFileRegex.match(entry):
            return fullPath

    return None

def deriveProteinIdFromFolder(folderName):
    if folderName.lower().endswith('_out'):
        return folderName[:-4]
    return folderName

def locatePocketAtmPath(proteinFolderPath, pocketId):
    relPath1 = os.path.join('pockets', f'pocket{pocketId}_atm.pdb')
    absPath1 = os.path.join(proteinFolderPath, relPath1)
    if os.path.isfile(absPath1):
        return absPath1, relPath1.replace('\\', '/')

    relPath2 = f'pocket{pocketId}_atm.pdb'
    absPath2 = os.path.join(proteinFolderPath, relPath2)
    if os.path.isfile(absPath2):
        return absPath2, relPath2.replace('\\', '/')

    return None, None

def workerBuildProteinEntry(proteinFolderPath):
    folderName = os.path.basename(proteinFolderPath.rstrip('/'))
    proteinId = deriveProteinIdFromFolder(folderName)

    proteinEntry = {
        'proteinID': proteinId,
        'proteinFolder': folderName,
        'infoFilePath': None,
        'pockets': {},
        'errors': []
    }

    try:
        infoFileAbsPath = findInfoFileInProteinFolder(proteinFolderPath, proteinId)
        if infoFileAbsPath is None:
            proteinEntry['errors'].append('Missing info file (*_info.txt).')
            jsonLine = json.dumps(proteinEntry, separators=(',', ':'), ensure_ascii=False) + '\n'
            return {'proteinID': proteinId, 'success': False, 'jsonLine': jsonLine, 'error': 'Missing info file'}

        proteinEntry['infoFilePath'] = os.path.basename(infoFileAbsPath)

        pocketStatsById = parseFpocketInfoFile(infoFileAbsPath)

        for pocketId, stats in pocketStatsById.items():
            pocketAtmAbsPath, pocketAtmRelPath = locatePocketAtmPath(proteinFolderPath, pocketId)

            pocketEntry = {
                'stats': stats,
                'aminoAcidRanges': [],
                'pocketAtmPdbPath': pocketAtmRelPath
            }

            if pocketAtmAbsPath is None:
                proteinEntry['errors'].append(f'Missing pocket structure file for pocket {pocketId} (expected pockets/pocket{pocketId}_atm.pdb).')
            else:
                try:
                    pocketEntry['aminoAcidRanges'] = buildAminoAcidRangesFromPocketAtm(pocketAtmAbsPath)
                except Exception as error:
                    proteinEntry['errors'].append(f'Failed parsing pocket {pocketId} atm PDB: {str(error)}')

            proteinEntry['pockets'][str(pocketId)] = pocketEntry

        jsonLine = json.dumps(proteinEntry, separators=(',', ':'), ensure_ascii=False) + '\n'
        success = (len(proteinEntry['errors']) == 0)
        return {'proteinID': proteinId, 'success': success, 'jsonLine': jsonLine, 'error': None if success else 'Completed with errors'}

    except Exception as error:
        proteinEntry['errors'].append(f'Unexpected error: {type(error).__name__}: {str(error)}')
        jsonLine = json.dumps(proteinEntry, separators=(',', ':'), ensure_ascii=False) + '\n'
        return {'proteinID': proteinId, 'success': False, 'jsonLine': jsonLine, 'error': str(error)}

if __name__ == '__main__':
    # Color logging
    LOG_LEVEL = logging.INFO
    LOGFORMAT = '%(log_color)s%(asctime)s - %(levelname)s - %(message)s'

    formatter = ColoredFormatter(
        LOGFORMAT,
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'white',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    )
    color_handler = logging.StreamHandler()
    color_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(f"Project Delta Logs/{os.path.basename(__file__)} {datetime.now().date()} {datetime.now().strftime('%H.%M.%S')}.txt")
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=LOG_LEVEL,
        format=LOGFORMAT,
        handlers=[file_handler, color_handler]
    )
    logging.info(f'{os.path.basename(__file__)}')
    logging.info(f'{datetime.now()}')
    logging.info('This file reads per-protein Fpocket outputs and produces a single JSON file containing every protein, every pocket, pocket statistics, amino-acid ranges, and the pocket structure path.')

    logging.info('Script started.')
    start_time = time.time()

    if not os.path.isdir(inputDirectory):
        logging.error(f'Input directory does not exist: {inputDirectory}')
        raise SystemExit(1)
    outputParentDir = os.path.dirname(outputFile)
    if outputParentDir and not os.path.isdir(outputParentDir):
        os.makedirs(outputParentDir, exist_ok=True)
    if os.path.isfile(outputFile):
        logging.warning(f'Output JSON file already exists. Overwriting: {outputFile}')
        os.remove(outputFile)

    proteinFolders = discoverProteinOutputFolders()
    totalProteins = len(proteinFolders)
    logging.info(f'Found {totalProteins} protein fpocket folders in {inputDirectory}')

    processedCount = 0
    failedCount = 0

    if totalProteins > 0:
        logging.info(f'Starting parallel folder parsing with {concurrentFolderParses} concurrent workers')

        with open(outputFile, 'w', encoding='utf-8') as outFile:
            with ThreadPoolExecutor(max_workers=concurrentFolderParses) as executor:
                futureToFolder = {executor.submit(workerBuildProteinEntry, folderPath): folderPath for folderPath in proteinFolders}

                for completedIndex, future in enumerate(as_completed(futureToFolder), start=1):
                    result = future.result()
                    proteinID = result['proteinID']

                    outFile.write(result['jsonLine'])

                    if result['success']:
                        processedCount += 1
                        logging.info(f'{completedIndex}/{totalProteins} | Indexed pockets for {proteinID}')
                    else:
                        failedCount += 1
                        logging.warning(f'{completedIndex}/{totalProteins} | Indexed {proteinID} with issues: {result["error"]}')

    # Final logs
    logging.info('JSON pocket index creation completed.')
    logging.info(f'Proteins indexed with no errors: {processedCount}')
    logging.info(f'Proteins indexed with warnings/errors: {failedCount}')
    end_time = time.time()
    total_time = round(end_time - start_time, 2)
    logging.info(f'Total time taken: {time.strftime("%H:%M:%S", time.gmtime(total_time))} seconds.')
