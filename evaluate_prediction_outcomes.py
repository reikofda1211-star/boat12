"""Compare archived pre-race 1-2 predictions with settled results.

Usage:
  python evaluate_prediction_outcomes.py --predictions two-stage-preview.csv \
      --results history/races.csv --output prediction_outcomes.json

Never re-trains a model. Keep the original prediction CSV as a timestamped
artifact; do not substitute retrospectively generated predictions.
"""
import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


def day(value):
    value = str(value).strip()
    for fmt in ('%Y-%m-%d', '%Y%m%d', '%Y/%m/%d'):
        try:
            return datetime.strptime(value, fmt).strftime('%Y%m%d')
        except ValueError:
            pass
    return None


def key(row):
    d = day(row.get('date', ''))
    venue = str(row.get('venue', '')).strip()
    try:
        race = int(str(row.get('race', '')).strip().upper().removesuffix('R'))
    except ValueError:
        return None
    if not d or not venue or not 1 <= race <= 12:
        return None
    return d, venue, race


def result_class(result):
    s = str(result).strip().replace('－', '-').replace('−', '-')
    m = re.fullmatch(r'([1-6])-([1-6])', s)
    if not m or m.group(1) == m.group(2):
        return None
    if s == '1-2':
        return 'hit_12'
    if m.group(1) == '1':
        return 'boat_2_missed_second'
    return 'boat_1_missed_first'


def probability(row, column):
    try:
        pct = float(row[column])
    except (KeyError, ValueError, TypeError):
        return None
    return pct / 100 if 0 <= pct <= 100 else None


def evaluate(predictions, results):
    actual = {}
    for row in results:
        k = key(row)
        c = result_class(row.get('result12', ''))
        if k and c:
            actual[k] = c
    counts = Counter()
    squared = Counter()
    duplicates = set()
    seen = set()
    details = []
    for row in predictions:
        k = key(row)
        if not k:
            counts['invalid_prediction_key'] += 1
            continue
        if k in seen:
            duplicates.add(k)
            continue
        seen.add(k)
        if k not in actual:
            counts['unmatched_or_unsettled'] += 1
            continue
        old = probability(row, 'direct_12_pct')
        new = probability(row, 'two_stage_12_pct')
        if old is None or new is None:
            counts['invalid_probability'] += 1
            continue
        label = actual[k]
        counts[label] += 1
        counts['matched'] += 1
        y = int(label == 'hit_12')
        squared['direct'] += (old-y)**2
        squared['two_stage'] += (new-y)**2
        details.append({'date':k[0], 'venue':k[1], 'race':k[2],
                        'outcome':label, 'direct_12_pct':round(old*100,2),
                        'two_stage_12_pct':round(new*100,2)})
    n = counts['matched']
    return {'matched':n, 'counts':dict(counts), 'duplicate_prediction_keys':len(duplicates),
            'hit_rate':round(counts['hit_12']/n,5) if n else None,
            'brier_direct':round(squared['direct']/n,6) if n else None,
            'brier_two_stage':round(squared['two_stage']/n,6) if n else None,
            'details':details,
            'caution':'This measures forecast accuracy only, not betting profitability. '
                      'Only evaluate archived forecasts created before race outcomes.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--results', default='history/races.csv', type=Path)
    parser.add_argument('--output', default='prediction_outcomes.json', type=Path)
    args = parser.parse_args()
    with args.predictions.open(encoding='utf-8-sig', newline='') as f:
        predictions = list(csv.DictReader(f))
    with args.results.open(encoding='utf-8-sig', newline='') as f:
        results = list(csv.DictReader(f))
    report = evaluate(predictions, results)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Saved {args.output}; matched={report['matched']}")


if __name__ == '__main__':
    main()
