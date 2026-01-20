### MSA to Mutation Mapping
### This file helps map mutations to MSA columns based on the presence of the mutation's different states in the MSA.

import os
import time
import json
import logging
import math
import sys
from datetime import datetime
from colorlog import ColoredFormatter

msaDirectory = '' # MSA Input Path
inputP2SNPFile = '' # Protein to Disease Mapping Input Path
outputFile = '' # Output Path

excludeQueryFromCounts = True
minOrthologs = 20
minP10CoverageOnQuery = 0.70
minFracQueryColsOccGe80 = 0.70

def safeMean(values):
    if not values:
        return None
    numeric = []
    for v in values:
        if isinstance(v, (int, float)) and (not math.isnan(v)):
            numeric.append(float(v))
    return (sum(numeric) / len(numeric)) if numeric else None

def safeQuantile(values, q):
    if not values:
        return None
    numeric = []
    for v in values:
        if isinstance(v, (int, float)) and (not math.isnan(v)):
            numeric.append(float(v))
    if not numeric:
        return None
    numeric = sorted(numeric)
    if len(numeric) == 1:
        return numeric[0]
    idx = int(round((len(numeric) - 1) * q))
    idx = max(0, min(len(numeric) - 1, idx))
    return numeric[idx]

def shannonEntropy(counts):
    total = 0
    for v in counts.values():
        total += v
    if total <= 0:
        return None
    ent = 0.0
    for v in counts.values():
        if v <= 0:
            continue
        p = v / total
        ent -= p * math.log2(p)
    return ent

def accessionFromHeader(header):
    return header.split()[0] if header else ''

def parseAlignedFasta(alnPath):
    sequences = []
    header = None
    buffer = []
    with open(alnPath, 'r', encoding='utf-8', errors='replace') as f:
        for rawLine in f:
            line = rawLine.rstrip('\n')
            if line.strip() == '':
                continue

            if line.startswith('>'):
                if header is not None:
                    sequences.append((header, ''.join(buffer)))
                header = line[1:].strip()
                buffer = []
            else:
                seqLine = line.strip().replace(' ', '').replace('\t', '')
                if seqLine != '':
                    buffer.append(seqLine)

    if header is not None:
        sequences.append((header, ''.join(buffer)))
    if not sequences:
        raise ValueError('No sequences found in alignment.')

    alnLen = len(sequences[0][1])
    for h, s in sequences:
        if len(s) != alnLen:
            raise ValueError(f'Non-rectangular alignment: {accessionFromHeader(h)} length {len(s)} != {alnLen}')

    return sequences

def pickQuerySequence(sequences, preferredAccession):
    pref = str(preferredAccession or '')
    prefBase = pref.split('.')[0]
    for i, (h, s) in enumerate(sequences):
        acc = accessionFromHeader(h)
        if acc == pref or acc.split('.')[0] == prefBase:
            return acc, s, i, (acc == pref)
    for i, (h, s) in enumerate(sequences):
        if 'homo sapiens' in h.lower():
            return accessionFromHeader(h), s, i, False
    return accessionFromHeader(sequences[0][0]), sequences[0][1], 0, False

def buildQueryPosToCol(querySeq):
    posToCol = {}
    pos = 0
    for col, ch in enumerate(querySeq):
        if ch != '-':
            pos += 1
            posToCol[pos] = col
    return posToCol, pos

def computeMSAStats(sequences, proteinID):
    queryAccession, querySeq, queryIndex, queryFoundExact = pickQuerySequence(sequences, proteinID)
    alignmentLength = len(querySeq)
    posToCol, queryUngappedLength = buildQueryPosToCol(querySeq)
    queryCols = [posToCol[pos] for pos in range(1, queryUngappedLength + 1)] if queryUngappedLength > 0 else []
    allIndices = list(range(len(sequences)))
    orthologIndices = [i for i in allIndices if (not excludeQueryFromCounts) or (i != queryIndex)]
    nOrthologsUsed = len(orthologIndices)

    overallGapFraction = None
    if nOrthologsUsed > 0 and alignmentLength > 0:
        totalGaps = 0
        for i in orthologIndices:
            s = sequences[i][1]
            totalGaps += sum(1 for ch in s if ch == '-')
        overallGapFraction = totalGaps / (nOrthologsUsed * alignmentLength)

    seqCoverageOnQuery = []
    identityToQuery = []

    for i in orthologIndices:
        s = sequences[i][1]
        nonGapCount = 0
        overlap = 0
        match = 0
        for col in queryCols:
            a = s[col]
            q = querySeq[col]
            if a != '-':
                nonGapCount += 1
            if a == '-' or q == '-':
                continue
            overlap += 1
            if a == q:
                match += 1
        seqCoverageOnQuery.append((nonGapCount / queryUngappedLength) if queryUngappedLength > 0 else 0.0)
        identityToQuery.append((match / overlap) if overlap > 0 else None)

    queryColOccupancy = []
    queryColEntropy = []

    if nOrthologsUsed > 0:
        for col in queryCols:
            nongap = 0
            counts = {}
            for i in orthologIndices:
                a = sequences[i][1][col]
                if a == '-':
                    continue
                nongap += 1
                counts[a] = counts.get(a, 0) + 1
            queryColOccupancy.append(nongap / nOrthologsUsed)
            queryColEntropy.append(shannonEntropy(counts))

    msaStats = {
        'queryAccessionUsed': queryAccession,
        'queryFoundExact': queryFoundExact,
        'nSeqTotal': len(sequences),
        'nOrthologsUsed': nOrthologsUsed,
        'alignmentLength': alignmentLength,
        'queryUngappedLength': queryUngappedLength,
        'alignmentExpansionRatio': (alignmentLength / queryUngappedLength) if queryUngappedLength > 0 else None,
        'overallGapFraction': overallGapFraction,
        'meanSeqCoverageOnQuery': safeMean(seqCoverageOnQuery),
        'p10SeqCoverageOnQuery': safeQuantile(seqCoverageOnQuery, 0.10),
        'fracSeqCoverageLt50': (sum(1 for x in seqCoverageOnQuery if x < 0.50) / len(seqCoverageOnQuery)) if seqCoverageOnQuery else None,
        'meanIdentityToQuery': safeMean(identityToQuery),
        'p10IdentityToQuery': safeQuantile(identityToQuery, 0.10),
        'meanQueryColOccupancy': safeMean(queryColOccupancy),
        'fracQueryColsOccupancyGe80': (sum(1 for x in queryColOccupancy if x >= 0.80) / len(queryColOccupancy)) if queryColOccupancy else None,
        'fracQueryColsOccupancyLt50': (sum(1 for x in queryColOccupancy if x < 0.50) / len(queryColOccupancy)) if queryColOccupancy else None,
        'meanShannonEntropyQueryCols': safeMean(queryColEntropy)
    }

    flags = []
    if (msaStats.get('nOrthologsUsed') or 0) < minOrthologs:
        flags.append('lowDepth')
    p10Cov = msaStats.get('p10SeqCoverageOnQuery')
    if p10Cov is not None and p10Cov < minP10CoverageOnQuery:
        flags.append('lowTailCoverage')
    fracOccGe80 = msaStats.get('fracQueryColsOccupancyGe80')
    if fracOccGe80 is not None and fracOccGe80 < minFracQueryColsOccGe80:
        flags.append('lowColumnSupport')

    msaQuality = {
        'passesThresholds': (len(flags) == 0),
        'flags': flags,
        'thresholds': {'minOrthologs': minOrthologs, 'minP10CoverageOnQuery': minP10CoverageOnQuery, 'minFracQueryColsOccGe80': minFracQueryColsOccGe80}
    }

    return msaStats, msaQuality, querySeq, queryIndex, posToCol

def loadP2SNPEntries(path):
    with open(path, 'r') as f:
        firstNonSpace = ''
        while True:
            ch = f.read(1)
            if ch == '':
                break
            if not ch.isspace():
                firstNonSpace = ch
                break
        f.seek(0)
        if firstNonSpace == '[':
            return json.load(f)
        entries = []
        for rawLine in f:
            line = rawLine.strip()
            if line == '':
                continue
            entries.append(json.loads(line))
        return entries

def extractSimpleSubstitutions(proteinEntry):
    mutations = []
    seen = set()

    for snp in proteinEntry.get('assocSNPsInfo', []):
        snpID = str(snp.get('SNPID', ''))
        varType = snp.get('VarType')

        pos = snp.get('position', None)
        deleted = snp.get('deleted_sequence', '')
        inserted = snp.get('inserted_sequence', '')

        if deleted == inserted:
            continue
        if not isinstance(pos, int):
            continue

        deleted = (str(deleted) if deleted is not None else '').upper()
        inserted = (str(inserted) if inserted is not None else '').upper()

        if len(deleted) != 1 or len(inserted) != 1:
            continue

        key = (snpID, pos, deleted, inserted)
        if key in seen:
            continue
        seen.add(key)

        mutations.append({
            'snpID': snpID,
            'varType': varType,
            'hgvs': snp.get('hgvs'),
            'pos0': pos,
            'position': pos + 1,
            'healthyAA': deleted,
            'mutatedAA': inserted
        })
        continue

    return mutations

def computeMutationConservation(sequences, queryIndex, querySeq, posToCol, mutation):
    pos1 = mutation.get('position')
    healthyAA = mutation.get('healthyAA')
    mutatedAA = mutation.get('mutatedAA')

    col = posToCol.get(pos1)
    if col is None:
        return {'status': 'positionNotInQueryAlignment', 'alignmentColumn0': None, 'queryResidue': None, 'matchesHealthyInQuery': None, 'counts': None, 'percentages': None, 'nOrthologsUsed': 0}

    indices = list(range(len(sequences)))
    if excludeQueryFromCounts:
        indices = [i for i in indices if i != queryIndex]

    counts = {'healthy': 0, 'mutated': 0, 'other': 0, 'gap': 0}
    for i in indices:
        aa = sequences[i][1][col]
        if aa == '-':
            counts['gap'] += 1
        elif aa == healthyAA:
            counts['healthy'] += 1
        elif aa == mutatedAA:
            counts['mutated'] += 1
        else:
            counts['other'] += 1

    denom = len(indices)
    percentages = {'healthy': (counts['healthy'] / denom) if denom > 0 else None, 'mutated': (counts['mutated'] / denom) if denom > 0 else None, 'other': (counts['other'] / denom) if denom > 0 else None, 'gap': (counts['gap'] / denom) if denom > 0 else None}
    queryResidue = querySeq[col] if col < len(querySeq) else None

    if queryResidue != healthyAA:
        return {'status': 'queryResidueMismatch', 'alignmentColumn0': col, 'queryResidue': queryResidue, 'matchesHealthyInQuery': False, 'counts': counts, 'percentages': percentages, 'nOrthologsUsed': denom}
    return {'status': 'ok', 'alignmentColumn0': col, 'queryResidue': queryResidue, 'matchesHealthyInQuery': True, 'counts': counts, 'percentages': percentages, 'nOrthologsUsed': denom}

def processProteinEntry(proteinEntry):
    proteinId = proteinEntry.get('protein')
    outEntry = {'proteinID': proteinId, 'numAssocSNPs': proteinEntry.get('numAssocSNPs'), 'msaStats': None, 'msaQuality': None, 'mutations': [], 'errors': []}

    if not proteinId:
        outEntry['errors'].append('Missing protein id in P2SNP entry.')
        return outEntry

    msaPath = os.path.join(msaDirectory, f'{proteinId}.aln.fasta')

    if not os.path.isfile(msaPath):
        outEntry['errors'].append(f'Missing MSA file: {msaPath}')
        return outEntry

    sequences = parseAlignedFasta(msaPath)
    msaStats, msaQuality, querySeq, queryIndex, posToCol = computeMSAStats(sequences, proteinId)
    outEntry['msaStats'] = msaStats
    outEntry['msaQuality'] = msaQuality

    mutations = extractSimpleSubstitutions(proteinEntry)
    for m in mutations:
        cons = computeMutationConservation(sequences, queryIndex, querySeq, posToCol, m)
        outEntry['mutations'].append({**m, **cons})

    return outEntry

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
    logging.info('This file helps map mutations to MSA columns based on the presence of the mutation\'s different states in the MSA.')

    logging.info('Script started.')
    start_time = time.time()

    if not os.path.isdir(msaDirectory):
        logging.error(f'MSA directory does not exist: {msaDirectory}')
        sys.exit(1)

    outputParentDir = os.path.dirname(outputFile)
    if outputParentDir and (not os.path.isdir(outputParentDir)):
        os.makedirs(outputParentDir, exist_ok=True)
    if os.path.isfile(outputFile):
        os.remove(outputFile)

    proteinEntries = loadP2SNPEntries(inputP2SNPFile)
    totalProteins = len(proteinEntries)
    logging.info(f'Loaded {totalProteins} protein entries from P2SNP')

    processedCount = 0
    failedCount = 0

    with open(outputFile, 'w') as outHandle:
        for i, proteinEntry in enumerate(proteinEntries, start=1):
            try:
                outEntry = processProteinEntry(proteinEntry)
                outHandle.write(json.dumps(outEntry, separators=(',', ':'), ensure_ascii=False) + '\n')

                if outEntry.get('errors'):
                    failedCount += 1
                    logging.warning(f'{i}/{totalProteins} | {outEntry.get("proteinID")} | {outEntry["errors"][0]}')
                else:
                    processedCount += 1
                    logging.info(f'{i}/{totalProteins} | Indexed {outEntry.get("proteinID")}')
            except Exception as error:
                failedCount += 1
                proteinID = proteinEntry.get('protein')
                outEntry = {'proteinID': proteinID, 'numAssocSNPs': proteinEntry.get('numAssocSNPs'), 'msaStats': None, 'msaQuality': None, 'mutations': [], 'errors': [f'Unexpected error: {type(error).__name__}: {str(error)}']}
                outHandle.write(json.dumps(outEntry, separators=(',', ':'), ensure_ascii=False) + '\n')
                logging.error(f'{i}/{totalProteins} | {proteinID} | {type(error).__name__}: {str(error)}')

    # Final logs
    logging.info('MSA to mutation mapping completed.')
    logging.info(f'Proteins indexed with no errors: {processedCount}')
    logging.info(f'Proteins indexed with warnings/errors: {failedCount}')
    end_time = time.time()
    total_time = round(end_time - start_time, 2)
    logging.info(f'Total time taken: {time.strftime("%H:%M:%S", time.gmtime(total_time))} seconds.')
    logging.info('---------------------------------------------------------------------------------------')
    logging.info("""
### Per-protein MSA statistics

# Query selection and trust
- queryAccessionUsed: Accession selected as the query sequence in the MSA. All position to column mapping uses this sequence.
- queryFoundExact: True if the query was found by exact accession match to proteinID. False means the script fell back to a Homo sapiens header or the first sequence; treat mapping/percentages with extra caution if False.

# Depth / size
- nSeqTotal: Total sequences in the MSA file (including the query).
- nOrthologsUsed: Number of sequences used as orthologs for stats and mutation counting (query excluded if excludeQueryFromCounts=True). Larger is generally better for stability.

# Alignment geometry
- alignmentLength: Number of alignment columns (including gaps).
- queryUngappedLength: Number of non-gap residues in the query within the alignment (this defines the 1-based coordinate system used for mapping position to alignmentColumn0).
- alignmentExpansionRatio: alignmentLength / queryUngappedLength. Higher values imply more gaps/indels relative to query; extremely high values often indicate lower-quality or highly variable alignments.

# Global gappiness
- overallGapFraction: Fraction of all characters across all ortholog sequences and all alignment columns that are gaps. High values usually mean fragmentary sequences and/or large indel regions.

# Per-ortholog sequence quality on query positions (computed only on query columns where the query has a residue)
- meanSeqCoverageOnQuery: Mean fraction of query positions covered (non-gap) by an ortholog. Higher means orthologs are more complete across the query protein.
- p10SeqCoverageOnQuery: 10th percentile of ortholog coverage. Low p10 indicates a bad tail of fragmentary orthologs (gaps will inflate site-level gap percentages).
- fracSeqCoverageLt50: Fraction of orthologs that cover less than 50% of query positions. High values mean many fragments.

# Similarity to the query (computed only on overlapping, non-gap positions on query columns)
- meanIdentityToQuery: Mean fraction of overlapping residues that exactly match the query residue.
- p10IdentityToQuery: 10th percentile identity. Extremely low values suggest a subset of very divergent sequences that may be harder to align reliably.

# Per-column support and variability (computed on query columns)
- meanQueryColOccupancy: Mean occupancy across query columns, where occupancy(col) = fraction of orthologs that are non-gap at that column. High occupancy means many orthologs actually cover most query sites.
- fracQueryColsOccupancyGe80: Fraction of query columns with occupancy >= 0.80. High values mean most sites are well-supported; low values mean many sites will have weak evidence.
- fracQueryColsOccupancyLt50: Fraction of query columns with occupancy < 0.50. High values mean many query sites are poorly covered.
- meanShannonEntropyQueryCols: Mean Shannon entropy across query columns, computed from non-gap residues only. Lower means more conserved columns overall; higher means more variable columns overall. Entropy is only meaningful when occupancy is not too low.

### Per-protein MSA quality summary
- passesThresholds: True if no quality flags triggered.
- flags:
  - lowDepth: nOrthologsUsed < minOrthologs. Percentages can be noisy with low depth.
  - lowTailCoverage: p10SeqCoverageOnQuery < minP10CoverageOnQuery. Many fragmentary sequences; site-level gap% may dominate.
  - lowColumnSupport: fracQueryColsOccupancyGe80 < minFracQueryColsOccGe80. Many query columns are not well-supported by most orthologs.
- thresholds: Records the threshold values used to create the flags.

### Per-mutation conservation statistics

# Mutation identity fields
- pos0: 0-based position in the reference protein sequence (SPDI position).
- position: 1-based position (pos0 + 1). This is mapped through the query alignment to a column.
- healthyAA: SPDI deleted_sequence (the reference/healthy amino acid).
- mutatedAA: SPDI inserted_sequence (the alternate/mutated amino acid).
- snpID, varType, hgvs: Identifiers and annotations for the variant.

# Mapping and validation
- status:
  - ok: position mapped to an alignment column.
  - positionNotInQueryAlignment: position is not present in the query’s ungapped alignment coordinates (query may be truncated/mismatched, or position is outside the aligned query).
  - queryResidueMismatch: queryResidue is different from healthyAA.
- alignmentColumn0: The 0-based alignment column used for residue counting at this site.
- queryResidue: The query’s aligned residue at that column.
- matchesHealthyInQuery: True if queryResidue == healthyAA. If False, treat the result with caution (possible isoform/version mismatch or wrong query selection).

# Counts and percentages across orthologs at the mutation column
- nOrthologsUsed: Denominator used for per-site counts/percentages (query excluded if configured).
- counts:
  - healthy: ortholog residue == healthyAA
  - mutated: ortholog residue == mutatedAA
  - other: ortholog residue is not gap and not equal to healthyAA or mutatedAA
  - gap: ortholog residue is a gap at that column
- percentages: counts divided by nOrthologsUsed. Note that gaps are included in the denominator; a high gap percentage means the site is poorly supported in the alignment.

# Practical interpretation for a mutation
- High healthy% with low mutated% and low other% (and low gap%) usually indicates strong conservation of the healthy residue; the mutation is more likely to disrupt function.
- High mutated% suggests the mutant residue occurs naturally in orthologs; the change may be more tolerated (context still matters).
- High other% suggests the site is variable; many substitutions are observed across orthologs.
- High gap% means many orthologs do not cover this site; treat the percentages as low-confidence evidence.
""")
