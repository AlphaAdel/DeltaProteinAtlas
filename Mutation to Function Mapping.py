### Mutation to Function Mapping
### This file combines the three sources of function data into one JSON file with a generated textual description added.

import os
import time
import json
import logging
from datetime import datetime
from colorlog import ColoredFormatter

impactScoresFilePath = '' # Mutation to Impact Mapping Input Path
msaMappingFilePath = '' # MSA to Mutation Mapping Input Path
proteinToPocketFilePath = '' # Protein to Pocket Mapping Input Path
outputFile = '' # Output Path

highImpactScoreMin = 0.8
moderateImpactScoreMin = 0.5
lowModerateImpactScoreMin = 0.3
highlyConservedHealthyMin = 0.90
conservedHealthyMin = 0.70
highGapMin = 0.35
variableOtherMin = 0.50
highDruggabilityMin = 0.5
moderateDruggabilityMin = 0.2
lowModerateDruggabilityMin = 0.05
nearPocketResidueRadius = 3
pocketStatsKeep = {
    'score',
    'druggabilityScore',
    'totalSasa',
    'polarSasa',
    'apolarSasa',
    'volume'
}

def roundScore4(x):
    if x is None:
        return None
    try:
        return round(float(x), 4)
    except Exception:
        return None

def formatScore4(x):
    v = roundScore4(x)
    if v is None:
        return None
    return f'{v:.4f}'

def formatFracAndPct4(fraction):
    if fraction is None:
        return None
    try:
        decimal = float(fraction)
    except Exception:
        return None
    return f'{decimal * 100.0:.2f}%'

def formatFloat3(x):
    if x is None:
        return None
    try:
        return f'{float(x):.3f}'
    except Exception:
        return None

def safeFloat(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s or s.lower() in {'n/a', 'na', 'none', 'null', 'nan'}:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None

def loadJSONRecords(filePath):
    if not os.path.isfile(filePath):
        raise FileNotFoundError(f'File not found: {filePath}')
    with open(filePath, 'r', encoding='utf-8') as f:
        text = f.read().strip()

    if not text:
        return []

    records = []
    for lineNumber, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f'JSON parse error in {filePath} at line {lineNumber}: {e}')
        if not isinstance(obj, dict):
            raise ValueError(f'Expected dict per line in {filePath}, got {type(obj)} at line {lineNumber}')
        records.append(obj)
    return records

def normalizeSnpID(snpID):
    if snpID is None:
        return ''
    s = str(snpID).strip()
    if s.lower().startswith('rs'):
        s = s[2:]
    digits = ''.join(ch for ch in s if ch.isdigit())
    return digits if digits else s

def normalizeAa(aa):
    if aa is None:
        return ''
    return str(aa).strip().upper()

def buildProteinMap(records, proteinKeyName):
    proteinMap = {}
    for rec in records:
        proteinID = str(rec.get(proteinKeyName, '')).strip()
        if not proteinID:
            continue
        proteinMap[proteinID] = rec
    return proteinMap

def getTopImpactDrivers(impactMethods):
    if not isinstance(impactMethods, dict):
        return []

    driverNames = ['atchley', 'kyteDoolittle', 'fractionalCharge', 'specialResidues']
    weightedPairs = []

    for name in driverNames:
        comp = impactMethods.get(name)
        w = None
        if isinstance(comp, dict):
            w = comp.get('weighted')
        elif isinstance(comp, (int, float)):
            w = comp
        if isinstance(w, (int, float)):
            weightedPairs.append((name, float(w)))

    if not weightedPairs:
        return []

    weightedPairs.sort(key=lambda x: x[1], reverse=True)
    topW = weightedPairs[0][1]
    if topW <= 0:
        return [weightedPairs[0][0]]

    topDrivers = []
    for name, w in weightedPairs:
        if (w / topW) >= 0.9:
            topDrivers.append(name)
    return topDrivers

def buildImpactComponentsWeighted(impactMethods):
    out = {}
    for name in ['atchley', 'kyteDoolittle', 'fractionalCharge', 'specialResidues']:
        w = None
        if isinstance(impactMethods, dict):
            comp = impactMethods.get(name)
            if isinstance(comp, dict):
                w = roundScore4(comp.get('weighted'))
            elif isinstance(comp, (int, float)):
                w = roundScore4(comp)
        out[name] = w
    return out

def classifyImpactSeverity(impactScore, varType, mutatedAa):
    mutatedAaNorm = normalizeAa(mutatedAa)
    varTypeStr = str(varType or '').lower()

    if mutatedAaNorm == '*':
        return 'very high', 'a stop codon introduces truncation, which can cause loss of function depending on position.'
    if 'frameshift' in mutatedAaNorm.lower() or 'frame' in varTypeStr:
        return 'very high', 'a frameshift alters the downstream sequence, typically destabilizing or truncating the protein.'
    if impactScore is None:
        return 'unknown', 'no numeric impactScore is available for this mutation type.'

    score = roundScore4(impactScore)
    if score is None:
        return 'unknown', 'impactScore could not be interpreted as a float.'

    if score >= highImpactScoreMin:
        return 'high', 'a large biochemical change occurs, likely to affect stability or interactions with other molecules.'
    if score >= moderateImpactScoreMin:
        return 'moderate', 'a moderate biochemical change occurs, but effects depend on conservation and structural context.'
    if score >= lowModerateImpactScoreMin:
        return 'low–moderate', 'a smaller biochemical change occurs, which matters at impactful functional sites.'
    return 'low', 'conservative substitution occurs, which is often tolerated unless at a critical residue.'

def rangesToStrings(aminoAcidRanges):
    if not isinstance(aminoAcidRanges, list):
        return []
    out = []
    for r in aminoAcidRanges:
        if not isinstance(r, dict):
            continue
        start = r.get('start')
        end = r.get('end')
        if start is None or end is None:
            continue
        try:
            out.append(f'{int(start)}-{int(end)}')
        except Exception:
            continue
    return out

def classifyConservation(msaMutation):
    if not msaMutation:
        return 'unavailable', 'no MSA conservation mapping was found for this mutation.'

    status = str(msaMutation.get('status') or '')
    if status and status != 'ok':
        return 'low confidence', f"alignment mapping status is '{status}', interpret conservation cautiously."

    pct = msaMutation.get('percentages')
    if not isinstance(pct, dict):
        return 'unavailable', 'no ortholog percentages are available.'

    healthy = float(pct.get('healthy') or 0.0)
    mutated = float(pct.get('mutated') or 0.0)
    other = float(pct.get('other') or 0.0)
    gap = float(pct.get('gap') or 0.0)

    if gap >= highGapMin:
        return 'indel/low-occupancy region', 'a high gap fraction suggests alignment uncertainty or flexible/inserted segment.'
    if healthy >= highlyConservedHealthyMin and other <= 0.08:
        return 'highly conserved', 'a healthy residue dominates across orthologs, so substitutions are more likely deleterious.'
    if healthy >= conservedHealthyMin and other <= 0.25:
        return 'conserved', 'the site is constrained, so changes may affect function or stability depending on structure.'
    if other >= variableOtherMin:
        return 'variable', 'many orthologs carry alternative residues, so the site likely tolerates substitutions more readily.'
    if mutated >= 0.10:
        return 'mutant-like residue observed', 'the specific mutant state appears in orthologs, which indicates partial tolerance.'

    return 'mixed', 'no single state strongly dominates, so it should be interpreted with structure and biochemical change.'

def positionInRange(pos1, start, end):
    return int(start) <= int(pos1) <= int(end)

def distanceToRange(pos1, start, end):
    pos1 = int(pos1)
    start = int(start)
    end = int(end)
    if start <= pos1 <= end:
        return 0
    return min(abs(pos1 - start), abs(pos1 - end))

def buildImpactDriversDetail(impact):
    if not isinstance(impact, dict):
        return None

    comps = impact.get('components')
    if not isinstance(comps, dict):
        return None

    weightedItems = []
    for name in ['fractionalCharge', 'atchley', 'kyteDoolittle', 'specialResidues']:
        w = roundScore4(comps.get(name))
        if w is None:
            continue
        weightedItems.append((w, name))
    if not weightedItems:
        return None
    weightedItems.sort(key=lambda x: x[0], reverse=True)

    topDrivers = impact.get('topDrivers') or []
    topDrivers = [str(x) for x in topDrivers if x]

    if topDrivers:
        selected = [(w, name) for (w, name) in weightedItems if name in topDrivers]
        if not selected:
            selected = weightedItems[:2]
    else:
        selected = weightedItems[:2]
    selected = selected[:3]

    outputNames = {'atchley': 'Atchley factors', 'kyteDoolittle': 'Kyte-Doolitle hydrophobicity scale', 'fractionalCharge': 'fractional charge', 'specialResidues': 'special components'}
    phrases = [f'the {outputNames[name]}, which contributes {w:.4f}' for (w, name) in selected]

    if len(phrases) == 1:
        return phrases[0]
    elif len(phrases) == 2:
        return ', and'.join(phrases)
    else:
        return '; '.join(phrases)

def buildConservationDetail(conservation):
    if not isinstance(conservation, dict):
        return None

    pct = conservation.get('percentages')
    nOrtho = conservation.get('nOrthologsUsed')
    status = conservation.get('status')

    if not isinstance(pct, dict):
        if status:
            return f"Alignment status is '{status}', so detailed ortholog percentages are unavailable."
        return None

    healthyP = formatFracAndPct4(pct.get('healthy'))
    mutatedP = formatFracAndPct4(pct.get('mutated'))
    otherP = formatFracAndPct4(pct.get('other'))
    gapP = formatFracAndPct4(pct.get('gap'))

    if None in [healthyP, mutatedP, otherP, gapP]:
        return 'Some or all analyses were not performed, so detailed ortholog percentages are unavailable.'

    return f'Across {nOrtho} orthologs, the healthy residue fraction is {healthyP} conserved, the mutant residue fraction is {mutatedP} conserved, other residues account for {otherP}, and gaps account for {gapP}.'

def shrinkPocketStats(stats):
    if not isinstance(stats, dict):
        return {}
    out = {}
    for k in pocketStatsKeep:
        if k not in stats:
            continue
        out[k] = safeFloat(stats.get(k))
    return out

def categorizeDruggability(drug):
    floatDrug = safeFloat(drug)
    if floatDrug is None:
        return 'because the druggability score is unavailable'
    if floatDrug >= highDruggabilityMin:
        return 'meaning this binding site is very likely to bind small molecules'
    if floatDrug >= moderateDruggabilityMin:
        return 'meaning this binding site is somewhat likely to bind small molecules'
    if floatDrug >= lowModerateDruggabilityMin:
        return 'meaning this binding site is unlikely to bind small molecules'
    return 'meaning this binding site is very unlikely to bind small molecules'

def formatPocketBrief(pocketMatch):
    if not isinstance(pocketMatch, dict):
        return None

    positions = ', '.join(pocketMatch.get('aminoAcidRanges'))
    stats = pocketMatch.get('stats') or {}
    if not isinstance(stats, dict):
        stats = {}
    vol = formatFloat3(stats.get('volume')) or 'N/A'
    sasa = formatFloat3(stats.get('totalSasa')) or 'N/A'

    drugVal = safeFloat(stats.get('druggabilityScore'))
    drug = formatScore4(drugVal) if drugVal is not None else 'N/A'

    return f'a binding site located at positions {positions} with volume {vol}, SASA {sasa}, and druggability {drug}, {categorizeDruggability(drug)}'

def findPocketMatchesForPos1(proteinPocketRecord, pos1):
    if not proteinPocketRecord:
        return [], None, []

    pockets = proteinPocketRecord.get('pockets')
    if not isinstance(pockets, dict):
        return [], None, []

    insideMatches = []
    nearestDistance = None
    nearestPockets = []

    for pocketID, pocketObj in pockets.items():
        ranges = pocketObj.get('aminoAcidRanges') or []
        bestDist = None
        isInside = False

        for r in ranges:
            start = r.get('start')
            end = r.get('end')
            if start is None or end is None:
                continue
            dist = distanceToRange(pos1, start, end)
            if bestDist is None or dist < bestDist:
                bestDist = dist
            if dist == 0:
                isInside = True

        if bestDist is None:
            continue

        pocketSummary = {
            'pocketID': str(pocketID),
            'stats': shrinkPocketStats(pocketObj.get('stats')),
            'aminoAcidRanges': rangesToStrings(pocketObj.get('aminoAcidRanges')),
            'pocketAtmPdbPath': pocketObj.get('pocketAtmPdbPath')
        }

        if isInside:
            insideMatches.append(pocketSummary)
            if nearestDistance is None or nearestDistance > 0:
                nearestDistance = 0
            continue

        if nearestDistance is None or bestDist < nearestDistance:
            nearestDistance = bestDist
            nearestPockets = [pocketSummary]
        elif bestDist == nearestDistance:
            nearestPockets.append(pocketSummary)

    return insideMatches, nearestDistance, nearestPockets

def findPocketMatchesForPos1List(proteinPocketRecord, pos1List):
    if not pos1List:
        return {
            'inPocket': False,
            'nearPocket': False,
            'nearestPocketDistance': None,
            'pocketMatchesByPos1': [],
            'pocketMatchesUnion': [],
            'nearestPocketMatchesByPos1': [],
            'nearestPocketMatchesUnion': []
        }

    pocketMatchesByPos1 = []
    nearestPocketMatchesByPos1 = []
    anyInPocket = False
    globalNearestDistance = None
    unionInsideByPocketId = {}
    nearestCandidatesByPos = []

    for pos1 in pos1List:
        insideMatches, nearestDist, nearestPockets = findPocketMatchesForPos1(proteinPocketRecord, pos1)

        pocketMatchesByPos1.append({
            'position': pos1,
            'pocketMatches': insideMatches,
            'nearestPocketDistance': nearestDist
        })

        nearestPocketMatchesByPos1.append({
            'position': pos1,
            'nearestPocketMatches': nearestPockets,
            'nearestPocketDistance': nearestDist
        })

        if insideMatches:
            anyInPocket = True
            for m in insideMatches:
                pocketId = m.get('pocketID')
                if pocketId:
                    unionInsideByPocketId[pocketId] = m

        if nearestDist is not None:
            if globalNearestDistance is None or nearestDist < globalNearestDistance:
                globalNearestDistance = nearestDist

        nearestCandidatesByPos.append((pos1, nearestDist, nearestPockets))

    unionNearestByPocketId = {}
    anyNearPocket = False

    if (not anyInPocket) and (globalNearestDistance is not None) and (globalNearestDistance > 0) and (globalNearestDistance <= nearPocketResidueRadius):
        anyNearPocket = True
        for _, nd, npockets in nearestCandidatesByPos:
            if nd == globalNearestDistance:
                for m in npockets or []:
                    pocketId = m.get('pocketID')
                    if pocketId:
                        unionNearestByPocketId[pocketId] = m

    return {
        'inPocket': anyInPocket,
        'nearPocket': anyNearPocket,
        'nearestPocketDistance': globalNearestDistance,
        'pocketMatchesByPos1': pocketMatchesByPos1,
        'pocketMatchesUnion': list(unionInsideByPocketId.values()),
        'nearestPocketMatchesByPos1': nearestPocketMatchesByPos1,
        'nearestPocketMatchesUnion': list(unionNearestByPocketId.values())
    }

def buildFunctionDescription(mergedMutation):
    impact = mergedMutation.get('impact') or {}
    impactScore = impact.get('impactScore')
    impactScoreStr = formatScore4(impactScore)
    severity = impact.get('severityLabel') or 'unknown'
    severityReason = impact.get('severityReason') or 'unavailable'

    conservation = mergedMutation.get('conservation') or {}
    conservationClass = conservation.get('conservationClass') or 'unavailable'
    conservationExplanation = conservation.get('conservationExplanation') or 'unavailable'
    conservationDetail = buildConservationDetail(conservation)

    pockets = mergedMutation.get('pockets') or {}
    inPocket = bool(pockets.get('inPocket'))
    nearPocket = bool(pockets.get('nearPocket'))
    nearestDistance = pockets.get('nearestPocketDistance')

    parts = []

    if severity in ['very high']:
        parts.append('The mutation has a very high likelihood of disrupting function.')
    elif severity in ['high'] and (conservationClass in ['highly conserved', 'conserved'] or inPocket):
        parts.append('The mutation has a high likelihood of functional impact.')
    elif severity in ['moderate'] and inPocket and conservationClass in ['highly conserved', 'conserved']:
        parts.append('The mutation has an elevated likelihood of functional impact in a conserved binding site context.')
    elif severity in ['low'] and conservationClass in ['highly conserved', 'conserved']:
        parts.append('The mutation\'s impact is possible despite conservative chemistry because the site is conserved.')
    else:
        parts.append('The mutation\'s functional impact is context-dependent and should be prioritized using conservation and structure.')

    if impactScoreStr is not None:
        parts.append(f'The mutation impact score is {impactScoreStr}, classified as {severity} because {severityReason}')
    else:
        parts.append(f'No mutation impact score is available, but severity is classified as {severity} because {severityReason}')

    driverInfo = buildImpactDriversDetail(impact)
    if driverInfo:
        parts.append(f'The mutation impact score is driven primarily by {driverInfo}.')

    parts.append(f'From an evolutionary conservation perspective, this site is classified as {conservationClass} because {conservationExplanation}')
    if conservationDetail:
        parts.append(conservationDetail)

    if inPocket:
        pocketMatches = pockets.get('pocketMatches') or []
        briefs = []
        for pm in pocketMatches:
            b = formatPocketBrief(pm)
            if b:
                briefs.append(b)
        if briefs:
            if len(briefs) == 1:
                parts.append(f'Structurally, the residue falls inside {briefs[0]}.')
            else:
                parts.append(f'Structurally, the residue falls inside: \n    - {'\n    - '.join(briefs)}')
        else:
            parts.append('Structurally, the residue falls inside a predicted binding site, but its statistics were unavailable.')
    elif nearPocket:
        nearMatches = pockets.get('nearestPocketMatches') or []
        briefs = []
        for pm in nearMatches:
            b = formatPocketBrief(pm)
            if b:
                briefs.append(b)
        if briefs:
            if len(briefs) == 1:
                parts.append(f'Structurally, the residue lies about {nearestDistance} residues from {briefs[0]}.')
            else:
                parts.append(f'Structurally, the residue lies about {nearestDistance} residues from: \n    - {'\n    - '.join(briefs)}')
        else:
            parts.append(f'Structurally, the residue lies within about {nearestDistance} residues of a predicted binding site, but its statistics were unavailable.')
    else:
        parts.append('Structurally, the residue does not map to any predicted binding site, so effects are more likely mediated through stability or other interaction interfaces.')

    return ' '.join(parts)

def mergeProteinRecords(proteinID, impactProteinRecord, msaProteinRecord, pocketProteinRecord):
    merged = {
        'proteinID': proteinID,
        'sourceCounts': {
            'impactNumAssocSnps': (impactProteinRecord or {}).get('numAssocSNPs'),
            'msaNumAssocSnps': (msaProteinRecord or {}).get('numAssocSnps') if isinstance((msaProteinRecord or {}).get('numAssocSnps'), int) else (msaProteinRecord or {}).get('numAssocSNPs'),
            'pocketErrors': (pocketProteinRecord or {}).get('errors')
        },
        'mutations': []
    }

    msaIndex = {}
    if msaProteinRecord and isinstance(msaProteinRecord.get('mutations'), list):
        for m in msaProteinRecord.get('mutations') or []:
            snpID = normalizeSnpID(m.get('snpID'))
            wtAa = normalizeAa(m.get('healthyAA'))
            mutAa = normalizeAa(m.get('mutatedAA'))

            pos1 = m.get('position')
            if pos1 is None and m.get('pos0') is not None:
                try:
                    pos1 = int(m.get('pos0')) + 1
                except Exception:
                    pos1 = None

            if pos1 is None:
                continue

            key = (snpID, wtAa, mutAa, int(pos1))

            existing = msaIndex.get(key)
            if existing is None:
                msaIndex[key] = m
            else:
                if str(existing.get('status')) != 'ok' and str(m.get('status')) == 'ok':
                    msaIndex[key] = m

    impactMutations = []
    if impactProteinRecord and isinstance(impactProteinRecord.get('assocSNPsInfo'), list):
        impactMutations = impactProteinRecord.get('assocSNPsInfo') or []

    mutationMap = {}

    for info in impactMutations:
        snpID = normalizeSnpID(info.get('SNPID'))
        wtAa = normalizeAa(info.get('impactWT'))
        mutAa = normalizeAa(info.get('impactMut'))

        positions0 = info.get('positions') or []
        if not isinstance(positions0, list):
            positions0 = []

        impactMethods = info.get('impactMethods') or {}
        topDrivers = getTopImpactDrivers(impactMethods)

        impactScoreRounded = roundScore4(info.get('impactScore'))

        severityLabel, severityReason = classifyImpactSeverity(impactScoreRounded, info.get('VarType'), info.get('impactMut'))

        for pos0 in positions0:
            if not isinstance(pos0, int):
                continue

            position = int(pos0) + 1
            key = (snpID, wtAa, mutAa, position)

            mutationMap[key] = {
                'snpID': snpID,
                'wtAa': wtAa,
                'mutAa': mutAa,
                'position': position,
                'impact': {
                    'impactScore': impactScoreRounded,
                    'impactNote': info.get('impactNote'),
                    'topDrivers': topDrivers,
                    'components': buildImpactComponentsWeighted(impactMethods),
                    'severityLabel': severityLabel,
                    'severityReason': severityReason
                },
                'conservation': None,
                'pockets': None
            }

    for key, m in msaIndex.items():
        if key in mutationMap:
            continue
        snpID, wtAa, mutAa, position = key

        mutationMap[key] = {
            'snpID': snpID,
            'wtAa': wtAa,
            'mutAa': mutAa,
            'position': int(position),
            'impact': {
                'impactScore': None,
                'impactNote': 'No mutation impact score record was found for this mutation in the impact file.',
                'topDrivers': [],
                'components': buildImpactComponentsWeighted({}),
                'severityLabel': 'Unknown',
                'severityReason': 'Missing mutation impact score record.'
            },
            'conservation': None,
            'pockets': None
        }

    for key, mergedMutation in mutationMap.items():
        snpID, wtAa, mutAa, position = key
        msaMutation = msaIndex.get(key)
        conservationClass, conservationExplanation = classifyConservation(msaMutation)

        pos1 = int(position)
        alignmentColumn0 = None
        pos1List = [pos1]
        if msaMutation:
            pos0 = msaMutation.get('pos0')
            pos1FromMsa = msaMutation.get('position')
            alignmentColumn0 = msaMutation.get('alignmentColumn0')
            if pos1FromMsa is not None:
                pos1 = int(pos1FromMsa)
                pos1List = [pos1]
        else:
            pos0 = pos1 - 1

        mergedMutation['conservation'] = {
            'status': (msaMutation or {}).get('status'),
            'pos0': pos0,
            'position': pos1,
            'alignmentColumn0': alignmentColumn0,
            'nOrthologsUsed': (msaMutation or {}).get('nOrthologsUsed'),
            'percentages': (msaMutation or {}).get('percentages'),
            'conservationClass': conservationClass,
            'conservationExplanation': conservationExplanation
        }

        pocketSummary = findPocketMatchesForPos1List(pocketProteinRecord, pos1List)

        mergedMutation['pockets'] = {
            'inPocket': pocketSummary.get('inPocket'),
            'nearPocket': pocketSummary.get('nearPocket'),
            'nearestPocketDistance': pocketSummary.get('nearestPocketDistance'),
            'pocketMatchesByPos1': pocketSummary.get('pocketMatchesByPos1'),
            'pocketMatches': pocketSummary.get('pocketMatchesUnion'),
            'nearestPocketMatchesByPos1': pocketSummary.get('nearestPocketMatchesByPos1'),
            'nearestPocketMatches': pocketSummary.get('nearestPocketMatchesUnion'),
            'proteinFolder': (pocketProteinRecord or {}).get('proteinFolder'),
            'infoFilePath': (pocketProteinRecord or {}).get('infoFilePath')
        }

        mergedMutation['functionDescription'] = buildFunctionDescription(mergedMutation)

    mergedMutationsList = list(mutationMap.values())
    mergedMutationsList.sort(key=lambda x: (int(x.get('snpID') or 0), x.get('wtAa') or '', x.get('mutAa') or '', int(x.get('position') or 0)))
    merged['mutations'] = mergedMutationsList

    if msaProteinRecord:
        merged['msaStats'] = msaProteinRecord.get('msaStats')
        merged['msaQuality'] = msaProteinRecord.get('msaQuality')
        merged['msaErrors'] = msaProteinRecord.get('errors')
    return merged

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
    logging.info('This files combines the three sources of function data into one JSON file with a generated textual description added.')

    logging.info('Script started.')
    start_time = time.time()

    impactRecords = loadJSONRecords(impactScoresFilePath)
    msaRecords = loadJSONRecords(msaMappingFilePath)
    pocketRecords = loadJSONRecords(proteinToPocketFilePath)
    logging.info(f'Impact records loaded: {len(impactRecords)}')
    logging.info(f'MSA records loaded: {len(msaRecords)}')
    logging.info(f'Pocket records loaded: {len(pocketRecords)}')

    logging.info('Building per-protein maps...')
    impactByProtein = buildProteinMap(impactRecords, 'protein')
    msaByProtein = buildProteinMap(msaRecords, 'proteinID')
    pocketByProtein = buildProteinMap(pocketRecords, 'proteinID')

    proteinIDs = set()
    proteinIDs.update(impactByProtein.keys())
    proteinIDs.update(msaByProtein.keys())
    proteinIDs.update(pocketByProtein.keys())
    proteinIDs = sorted(proteinIDs)
    logging.info(f'Total unique proteins to merge: {len(proteinIDs)}')

    proteinEntries = []
    for i, proteinID in enumerate(proteinIDs, start=1):
        logging.info(f'Merging protein {i}/{len(proteinIDs)}: {proteinID}')
        entry = mergeProteinRecords(proteinID, impactByProtein.get(proteinID), msaByProtein.get(proteinID), pocketByProtein.get(proteinID))
        proteinEntries.append(entry)

    with open(outputPath, 'w') as outfile:
        for entry in proteinEntries:
            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
    logging.info(f'Wrote {len(proteinEntries)} protein entries.')

    # Final logs
    logging.info('Mutation function generation complete.')
    end_time = time.time()
    total_time = round(end_time - start_time, 2)
    logging.info(f'Total time taken: {time.strftime("%H:%M:%S", time.gmtime(total_time))}.')
