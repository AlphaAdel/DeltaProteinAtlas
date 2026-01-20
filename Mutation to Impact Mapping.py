### Mutation to Impact Mapping
### This file incorporates several properties of amino acids to determine the impact of an amino acid mutation with a score from 0 to 1.

import os
import sys
import time
import numpy as np
import json
import logging
from datetime import datetime
from colorlog import ColoredFormatter
from concurrent.futures import ThreadPoolExecutor, as_completed

inputFilePath = '' # Input Path
outputFile = '' # Output Path

pH = 7.4
wAtchley = 0.45
wKyteDoolittle = 0.20
wCharge = 0.25
wSpecial = 0.10

concurrentProteinWorkers = 14
batchSizeProteins = 375

validAA = set(list('ACDEFGHIKLMNPQRSTVWY*'))
canonicalAA = set(list('ACDEFGHIKLMNPQRSTVWY'))
acidicSideChains = set(list('DECY'))
basicSideChains = set(list('HKR'))

atchleyFactors = {
    "A": (-0.591, -1.302, -0.733, 1.570, -0.146),
    "C": (-1.343, 0.465, -0.862, -1.020, -0.255),
    "D": (1.050, 0.302, -3.656, -0.259, -3.242),
    "E": (1.357, -1.453, 1.477, 0.113, -0.837),
    "F": (-1.006, -0.590, 1.891, -0.397, 0.412),
    "G": (-0.384, 1.652, 1.330, 1.045, 2.064),
    "H": (0.336, -0.417, -1.673, -1.474, -0.078),
    "I": (-1.239, -0.547, 2.131, 0.393, 0.816),
    "K": (1.831, -0.561, 0.533, -0.277, 1.648),
    "L": (-1.019, -0.987, -1.505, 1.266, -0.912),
    "M": (-0.663, -1.524, 2.219, -1.005, 1.212),
    "N": (0.945, 0.828, 1.299, -0.169, 0.933),
    "P": (0.189, 2.081, -1.628, 0.421, -1.392),
    "Q": (0.931, -0.179, -3.005, -0.503, -1.853),
    "R": (1.538, -0.055, 1.502, 0.440, 2.897),
    "S": (-0.228, 1.399, -4.760, 0.670, -2.647),
    "T": (-0.032, 0.326, 2.213, 0.908, 1.313),
    "V": (-1.337, -0.279, -0.544, 1.242, -1.262),
    "W": (-0.595, 0.009, 0.672, -2.128, -0.184),
    "Y": (0.260, 0.830, 3.097, -0.838, 1.512)
}
atchleyFactorsNp = {}
for aa, vec in atchleyFactors.items():
    atchleyFactorsNp[aa] = np.array(vec, dtype=np.float64)

kyteDoolittle = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5,
    "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8,
    "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5,
    "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5
}
pKaTable = {"D": 3.5, "E": 4.2, "H": 6.6, "C": 6.8, "Y": 10.3, "K": 10.5, "R": 12.5}

def safeStrip(value):
    if value is None:
        return ''
    return str(value).strip()

def isSingleAAToken(value):
    s = safeStrip(value)
    return (len(s) == 1) and (s in validAA)

def euclideanDistance5D(aaA, aaB):
    diff = atchleyFactorsNp[aaA] - atchleyFactorsNp[aaB]
    return float(np.sqrt(np.sum(diff * diff)))

def getFractionalSideChainCharge(aa, pHValue):
    if aa not in pKaTable:
        return 0.0
    pKa = pKaTable[aa]
    if aa in acidicSideChains:
        denom = 1.0 + float(np.power(10.0, (pKa - pHValue)))
        fracDeprot = 1.0 / denom
        return -fracDeprot
    if aa in basicSideChains:
        denom = 1.0 + float(np.power(10.0, (pHValue - pKa)))
        fracProt = 1.0 / denom
        return +fracProt
    return 0.0

def getSpecialResidueComponent(wtAA, mutAA):
    score = 0.0
    if wtAA == 'G' or mutAA == 'G':
        score += 0.35
    if wtAA == 'P' or mutAA == 'P':
        score += 0.35
    if wtAA == 'C' or mutAA == 'C':
        score += 0.25
    if (wtAA == 'C') ^ (mutAA == 'C'):
        score += 0.15
    if score > 1.0:
        score = 1.0
    return score

def computeMaxAtchleyDistance():
    aaList = sorted(list(canonicalAA))
    maxDist = 0.0
    for i in range(len(aaList)):
        for j in range(i + 1, len(aaList)):
            a = aaList[i]
            b = aaList[j]
            d = euclideanDistance5D(a, b)
            if d > maxDist:
                maxDist = d
    if maxDist <= 0:
        maxDist = 1.0
    return float(maxDist)

def computeMaxKyteDoolittleDelta():
    aaList = sorted(list(canonicalAA))
    maxDelta = 0.0
    for i in range(len(aaList)):
        for j in range(i + 1, len(aaList)):
            a = aaList[i]
            b = aaList[j]
            d = float(np.abs(kyteDoolittle[a] - kyteDoolittle[b]))
            if d > maxDelta:
                maxDelta = d
    if maxDelta <= 0:
        maxDelta = 1.0
    return float(maxDelta)

def computeMaxChargeDelta(pHValue):
    aaList = sorted(list(canonicalAA))
    maxDelta = 0.0
    for i in range(len(aaList)):
        for j in range(i + 1, len(aaList)):
            a = aaList[i]
            b = aaList[j]
            qa = getFractionalSideChainCharge(a, pHValue)
            qb = getFractionalSideChainCharge(b, pHValue)
            d = float(np.abs(qb - qa))
            if d > maxDelta:
                maxDelta = d
    if maxDelta <= 0:
        maxDelta = 1.0
    return float(maxDelta)

def normalizeWeights():
    total = float(wAtchley + wKyteDoolittle + wCharge + wSpecial)
    if total <= 0:
        return 0.45, 0.20, 0.25, 0.10
    return float(wAtchley / total), float(wKyteDoolittle / total), float(wCharge / total), float(wSpecial / total)

def buildCondenseKey(mutationObj):
    snpID = safeStrip(mutationObj.get('SNPID', None))
    varType = safeStrip(mutationObj.get('VarType', None))
    wtAA = safeStrip(mutationObj.get('deleted_sequence', None))
    mutAA = safeStrip(mutationObj.get('inserted_sequence', None))
    return snpID, varType, wtAA, mutAA

def sortPositions(positionValues):
    if not positionValues:
        return []
    numeric = []
    nonNumeric = []
    for v in positionValues:
        if v is None:
            continue
        if isinstance(v, int):
            numeric.append(v)
            continue
        s = safeStrip(v)
        if s == '':
            continue
        try:
            numeric.append(int(s))
        except Exception:
            nonNumeric.append(s)
    numeric = sorted(set(numeric))
    nonNumeric = sorted(set(nonNumeric))
    if nonNumeric:
        return numeric + nonNumeric
    return numeric

def condenseMutationsByKey(mutations):
    groups = {}
    for mutationObj in mutations:
        if not isinstance(mutationObj, dict):
            continue
        key = buildCondenseKey(mutationObj)
        posVal = mutationObj.get('position', None)

        if key not in groups:
            baseObj = dict(mutationObj)
            baseObj.pop('position', None)
            baseObj['positions'] = []
            groups[key] = {'obj': baseObj, 'posList': []}
        else:
            diseasesNew = mutationObj.get('diseases', None)
            if isinstance(diseasesNew, list):
                diseasesBase = groups[key]['obj'].get('diseases', None)
                if not isinstance(diseasesBase, list):
                    groups[key]['obj']['diseases'] = []
                    diseasesBase = groups[key]['obj']['diseases']
                for disease in diseasesNew:
                    if disease not in diseasesBase:
                        diseasesBase.append(disease)
        groups[key]['posList'].append(posVal)

    condensed = []
    for key in sorted(groups.keys()):
        baseObj = groups[key]['obj']
        baseObj['positions'] = sortPositions(groups[key]['posList'])
        condensed.append(baseObj)
    return condensed

def setUnscorableMutationFields(mutationObj, wtAA, mutAA, noteText):
    mutationObj['impactWT'] = wtAA
    mutationObj['impactMut'] = mutAA
    mutationObj['impactScore'] = None
    mutationObj['impactNote'] = noteText
    mutationObj['impactMethods'] = {
        'atchley': {'distance': None, 'normalized': None, 'weighted': None},
        'kyteDoolittle': {'delta': None, 'normalized': None, 'weighted': None},
        'fractionalCharge': {'wt': None, 'mut': None, 'delta': None, 'normalized': None, 'weighted': None},
        'specialResidues': {'component': None, 'weighted': None}
    }

def setScoredMutationFields(mutationObj, wtAA, mutAA, score, impactMethods):
    mutationObj['impactWT'] = wtAA
    mutationObj['impactMut'] = mutAA
    mutationObj['impactScore'] = score
    mutationObj.pop('impactNote', None)
    mutationObj['impactMethods'] = impactMethods

def computeImpactForMissense(wtAA, mutAA, maxAtchley, maxKd, maxCharge, wA, wK, wC, wS):
    atchleyDist = euclideanDistance5D(wtAA, mutAA)
    kdDelta = float(np.abs(kyteDoolittle[wtAA] - kyteDoolittle[mutAA]))
    qWt = getFractionalSideChainCharge(wtAA, pH)
    qMut = getFractionalSideChainCharge(mutAA, pH)
    chargeDelta = float(np.abs(qMut - qWt))
    specialComponent = getSpecialResidueComponent(wtAA, mutAA)
    atchleyNorm = float(atchleyDist / maxAtchley)
    kdNorm = float(kdDelta / maxKd)
    chargeNorm = float(chargeDelta / maxCharge)
    raw = (wA * atchleyNorm) + (wK * kdNorm) + (wC * chargeNorm) + (wS * specialComponent)
    score = float(np.clip(raw, 0.0, 1.0))
    impactMethods = {
        'atchley': {
            'distance': float(atchleyDist),
            'normalized': float(atchleyNorm),
            'weighted': float(wA * atchleyNorm)
        },
        'kyteDoolittle': {
            'delta': float(kdDelta),
            'normalized': float(kdNorm),
            'weighted': float(wK * kdNorm)
        },
        'fractionalCharge': {
            'wt': float(qWt),
            'mut': float(qMut),
            'delta': float(chargeDelta),
            'normalized': float(chargeNorm),
            'weighted': float(wC * chargeNorm)
        },
        'specialResidues': {
            'component': float(specialComponent),
            'weighted': float(wS * specialComponent)
        }
    }
    return score, impactMethods

def scoreMutationObject(mutationObj, maxAtchley, maxKd, maxCharge, wA, wK, wC, wS):
    wtAA = safeStrip(mutationObj.get('deleted_sequence', None))
    mutAA = safeStrip(mutationObj.get('inserted_sequence', None))
    insertedLower = mutAA.lower()
    if insertedLower == 'frameshift':
        setUnscorableMutationFields(mutationObj, wtAA, mutAA, 'Unscorable with Atchley/Kyte-Doolittle/pKa model: Frameshift mutation.')
        return 'frameshift'
    if wtAA == '*' or mutAA == '*':
        setUnscorableMutationFields(mutationObj, wtAA, mutAA, 'Unscorable with Atchley/Kyte-Doolittle/pKa model: Stop codon (*) mutation.')
        return 'stop'
    if (not isSingleAAToken(wtAA)) or (not isSingleAAToken(mutAA)):
        setUnscorableMutationFields(mutationObj, wtAA, mutAA, 'Unscorable with Atchley/Kyte-Doolittle/pKa model: Deleted or inserted sequence is not a single amino acid.')
        return 'nonSingle'
    if wtAA == mutAA:
        impactMethods = {
            'atchley': {'distance': 0.0, 'normalized': 0.0, 'weighted': 0.0},
            'kyteDoolittle': {'delta': 0.0, 'normalized': 0.0, 'weighted': 0.0},
            'fractionalCharge': {'wt': float(getFractionalSideChainCharge(wtAA, pH)), 'mut': float(getFractionalSideChainCharge(mutAA, pH)), 'delta': 0.0, 'normalized': 0.0, 'weighted': 0.0},
            'specialResidues': {'component': 0.0, 'weighted': 0.0}, 'ruleApplied': 'identity'
        }
        setScoredMutationFields(mutationObj, wtAA, mutAA, 0.0, impactMethods)
        return 'identity'
    if wtAA not in canonicalAA or mutAA not in canonicalAA:
        setUnscorableMutationFields(mutationObj, wtAA, mutAA, 'Unscorable: residue not in the canonical amino-acid set for Atchley/KD tables.')
        return 'nonCanonical'
    score, impactMethods = computeImpactForMissense(wtAA, mutAA, maxAtchley, maxKd, maxCharge, wA, wK, wC, wS)
    setScoredMutationFields(mutationObj, wtAA, mutAA, score, impactMethods)
    return 'scored'

def processProteinRecord(proteinRecord, maxAtchley, maxKd, maxCharge, wA, wK, wC, wS):
    proteinID = proteinRecord.get('protein', None)
    mutations = proteinRecord.get('assocSNPsInfo', None)
    if mutations is None or not isinstance(mutations, list):
        proteinRecord['impactError'] = 'Missing or invalid mutations list'
        return {
            'protein': proteinID,
            'proteinsProcessed': 1,
            'mutationsTotalInput': 0,
            'mutationsTotalCondensed': 0,
            'mutationsTotal': 0,
            'scored': 0,
            'frameshift': 0,
            'stop': 0,
            'nonSingle': 0,
            'identity': 0,
            'nonCanonical': 0
        }
    inputCount = len(mutations)
    condensedMutations = condenseMutationsByKey(mutations)
    proteinRecord['assocSNPsInfo'] = condensedMutations
    proteinRecord['numAssocSNPs'] = len(condensedMutations)
    condensedCount = len(condensedMutations)
    counts = {
        'protein': proteinID,
        'proteinsProcessed': 1,
        'mutationsTotalInput': inputCount,
        'mutationsTotalCondensed': condensedCount,
        'mutationsTotal': 0,
        'scored': 0,
        'frameshift': 0,
        'stop': 0,
        'nonSingle': 0,
        'identity': 0,
        'nonCanonical': 0
    }
    for mutationObj in condensedMutations:
        if not isinstance(mutationObj, dict):
            continue
        counts['mutationsTotal'] += 1
        status = scoreMutationObject(mutationObj, maxAtchley, maxKd, maxCharge, wA, wK, wC, wS)
        if status in counts:
            counts[status] += 1
    proteinRecord.pop('impactScoringInfo', None)
    proteinRecord['impactSummary'] = counts
    return counts

def processProteinLine(line, maxAtchley, maxKd, maxCharge, wA, wK, wC, wS):
    proteinRecord = json.loads(line)
    if proteinRecord is None or not isinstance(proteinRecord, dict):
        return None, None
    counts = processProteinRecord(proteinRecord, maxAtchley, maxKd, maxCharge, wA, wK, wC, wS)
    return proteinRecord, counts

def processBatch(batchLines, executor, maxAtchley, maxKd, maxCharge, wA, wK, wC, wS):
    futures = {}
    for index, line in batchLines:
        future = executor.submit(processProteinLine, line, maxAtchley, maxKd, maxCharge, wA, wK, wC, wS)
        futures[future] = index
    results = {}
    countsByIndex = {}
    for future in as_completed(futures):
        index = futures[future]
        try:
            proteinRecord, counts = future.result()
            results[index] = proteinRecord
            countsByIndex[index] = counts
        except Exception as e:
            results[index] = None
            countsByIndex[index] = None
            logging.exception(f'Worker failed for protein line index={index}: {e}')
    return results, countsByIndex

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
    logging.info('This file incorporates several properties of amino acids to determine the impact of an amino acid mutation with a score from 0 to 1.')

    logging.info('Script started.')
    start_time = time.time()

    if not os.path.isfile(inputFilePath):
        logging.error(f'Input file does not exist: {inputFilePath}')
        sys.exit(1)

    wA, wK, wC, wS = normalizeWeights()
    maxAtchley = computeMaxAtchleyDistance()
    maxKd = computeMaxKyteDoolittleDelta()
    maxCharge = computeMaxChargeDelta(pH)

    logging.info(f'pH: {pH}')
    logging.info(f'Normalized weights | Atchley: {wA}, Kyte-Doolittle: {wK}, Charge: {wC}, Special: {wS}')
    logging.info(f'Normalization max | Atchley: {maxAtchley}, Kyte-Doolittle: {maxKd}, Charge: {maxCharge}')

    proteinsProcessed = 0
    mutationsTotal = 0
    scoredTotal = 0
    frameshiftTotal = 0
    stopTotal = 0
    nonSingleTotal = 0
    identityTotal = 0
    nonCanonicalTotal = 0

    batchLines = []
    lineIndex = 0
    with ThreadPoolExecutor(max_workers=concurrentProteinWorkers) as executor:
        with open(inputFilePath, 'r', encoding='utf-8') as infile, open(outputFilePath, 'w', encoding='utf-8') as outfile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                batchLines.append((lineIndex, line))
                lineIndex += 1

                if len(batchLines) < batchSizeProteins:
                    continue

                results, countsByIndex = processBatch(batchLines, executor, maxAtchley, maxKd, maxCharge, wA, wK, wC, wS)

                for index in sorted(results.keys()):
                    proteinRecord = results[index]
                    counts = countsByIndex[index]
                    if proteinRecord is None or counts is None:
                        logging.warning(f'Skipping invalid JSON line {index}.')
                        continue
                    outfile.write(json.dumps(proteinRecord))
                    outfile.write('\n')

                    proteinsProcessed += 1
                    mutationsTotal += counts.get('mutationsTotal', 0)
                    scoredTotal += counts.get('scored', 0)
                    frameshiftTotal += counts.get('frameshift', 0)
                    stopTotal += counts.get('stop', 0)
                    nonSingleTotal += counts.get('nonSingle', 0)
                    identityTotal += counts.get('identity', 0)
                    nonCanonicalTotal += counts.get('nonCanonical', 0)

                    logging.info(f'{proteinsProcessed:,} processed proteins | {mutationsTotal:,} processed mutations | {scoredTotal:,} scored mutations | {stopTotal:,} unscored stop codon mutations | {frameshiftTotal:,} unscored frameshift mutations | {nonSingleTotal:,} unscored non-single AA mutations | {identityTotal:,} identity mutations | {nonCanonicalTotal:,} non-canonical AA mutations')

                batchLines = []

            if len(batchLines) > 0:
                results, countsByIndex = processBatch(batchLines, executor, maxAtchley, maxKd, maxCharge, wA, wK, wC, wS)

                for index in sorted(results.keys()):
                    proteinRecord = results[index]
                    counts = countsByIndex[index]
                    if proteinRecord is None or counts is None:
                        logging.warning(f'Skipping invalid JSON line {index}.')
                        continue
                    outfile.write(json.dumps(proteinRecord))
                    outfile.write('\n')

                    proteinsProcessed += 1
                    mutationsTotal += counts.get('mutationsTotal', 0)
                    scoredTotal += counts.get('scored', 0)
                    frameshiftTotal += counts.get('frameshift', 0)
                    stopTotal += counts.get('stop', 0)
                    nonSingleTotal += counts.get('nonSingle', 0)
                    identityTotal += counts.get('identity', 0)
                    nonCanonicalTotal += counts.get('nonCanonical', 0)

    # Final logs
    logging.info('Impact scoring complete.')
    logging.info(f'Final counts | {proteinsProcessed:,} total processed proteins | {mutationsTotal:,} total processed mutations | {scoredTotal:,} total scored mutations | {stopTotal:,} total unscored stop codon mutations | {frameshiftTotal:,} total unscored frameshift mutations | {nonSingleTotal:,} total unscored non-single AA mutations | {identityTotal:,} total identity mutations | {nonCanonicalTotal:,} total non-canonical AA mutations')
    end_time = time.time()
    total_time = round(end_time - start_time, 2)
    logging.info(f'Total time taken: {time.strftime("%H:%M:%S", time.gmtime(total_time))}.')
