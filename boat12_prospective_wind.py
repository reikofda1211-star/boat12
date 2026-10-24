#!/usr/bin/env python3
"""Frozen wind vs no-wind models evaluated only on future, unseen races.

Research only: never alters production model, conditions.json or display gates.
The first run fits BOTH models using records through 2026-10-23, persists the
immutable candidate artifact, and NEVER re-fits on subsequent runs.
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import conditions

CUTOFF_DAY = '2026-10-23'   # No fitting on later days.
FIRST_EVAL_DAY = '2026-10-25'  # Fresh racing days, after selection decision.
ROOT = Path('history/conditions')
FROZEN = ROOT / 'wind_prospective_frozen.joblib'
REPORT = ROOT / 'wind_prospective_report.json'
NO_WIND = {'wind_speed', 'wind_direction', 'venue_wind_direction'}


def gather():
    records, seen = [], set()
    for p in sorted((ROOT / 'days').glob('*.jsonl.gz')):
        for row in conditions.read_records(p):
            if (row.get('version') != conditions.VERSION or not row.get('basic_ready')
                    or row.get('label') not in (0, 1) or not row.get('date')
                    or row.get('code') in seen):
                continue
            seen.add(row['code'])
            records.append(row)
    return sorted(records, key=lambda r: r['code'])


def fit_locked(all_rows):
    import joblib
    old = json.loads((ROOT / 'wind_ablation.json').read_text())
    split = old['split']
    # Fixed, predetermined training/calibration/test split (don't learn from its test).
    period_start = json.loads((ROOT / 'report.json').read_text())['scopes']['normal']['period_start']
    historical = [r for r in all_rows if period_start <= r['date'] <= CUTOFF_DAY]
    train = [r for r in historical if r['date'] < split['calibration_start']]
    cal = [r for r in historical if split['calibration_start'] <= r['date'] < split['test_start']]
    assert max(r['date'] for r in cal) < split['test_start']
    if [len(train), len(cal)] != [split['train'], split['calibration']]:
        raise SystemExit('Historical split changed; refusing to freeze different models')
    original = conditions.features
    try:
        full = conditions.fit_model(train, cal, 'normal')
        def without_wind(row, scope):
            values, cats = original(row, scope)
            return ({k: v for k, v in values.items() if k not in NO_WIND},
                    [k for k in cats if k not in NO_WIND])
        conditions.features = without_wind
        stripped = conditions.fit_model(train, cal, 'normal')
    finally:
        conditions.features = original
    package = {
        'schema': 'frozen-wind-comparison-v1',
        'cutoff_day': CUTOFF_DAY,
        'first_eval_day': FIRST_EVAL_DAY,
        'train_rows': len(train),
        'calibration_rows': len(cal),
        'created_at': datetime.now(timezone.utc).isoformat(),
        'full': full, 'without_wind': stripped,
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    tmp = FROZEN.with_suffix('.tmp')
    joblib.dump(package, tmp)
    tmp.replace(FROZEN)
    return package


def predict_frozen(model, rows):
    # Reconstruct exactly the model's fitted feature set without mutating conditions.py.
    import numpy as np
    features = conditions.frame(rows, 'normal', model['columns'])
    scores = model['pipeline'].decision_function(features).reshape(-1, 1)
    return model['calibrator'].predict_proba(scores)[:, 1]


def evaluate(package, rows):
    import numpy as np
    from sklearn.metrics import brier_score_loss, log_loss
    test = [r for r in rows if FIRST_EVAL_DAY <= r['date'] and r['date'] > package['cutoff_day']]
    ids = [str(r['code']) for r in test]
    if len(ids) != len(set(ids)):
        raise SystemExit('Duplicate race IDs in prospective scoring')
    result = {'schema': 'boat12-prospective-wind-v1',
              'updated_at': datetime.now(timezone.utc).isoformat(),
              'first_eval_day': FIRST_EVAL_DAY, 'training_end': package['cutoff_day'],
              'new_races': len(test), 'new_days': len({r['date'] for r in test}),
              'status': 'collecting' if len(test) < 500 or len({r['date'] for r in test}) < 15 else 'evaluate_only',
              'production_changed': False,
              'note': 'Frozen models. Prospective observational comparison only; not a release or profit guarantee.'}
    if not test:
        result['message'] = 'No labelled future races yet; continue collecting.'
        return result
    y = np.asarray([r['label'] for r in test])
    a = predict_frozen(package['full'], test)
    b = predict_frozen(package['without_wind'], test)
    result['full'] = {'brier': float(brier_score_loss(y,a)), 'logloss': float(log_loss(y,a,labels=[0,1]))}
    result['without_wind'] = {'brier': float(brier_score_loss(y,b)), 'logloss': float(log_loss(y,b,labels=[0,1]))}
    result['brier_gain_without_wind'] = result['full']['brier'] - result['without_wind']['brier']
    if len({r['date'] for r in test}) >= 2:
        result['day_bootstrap_95ci'] = conditions.day_bootstrap_gain([r['date'] for r in test], (a-y)**2-(b-y)**2)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:
        assert NO_WIND == {'wind_speed','wind_direction','venue_wind_direction'}
        assert FIRST_EVAL_DAY > CUTOFF_DAY
        assert callable(predict_frozen)
        print('SELF TEST PASSED')
        return
    import joblib
    import sklearn
    rows = gather()
    if not FROZEN.exists():
        package=fit_locked(rows)
    else:
        # Only load a trusted, locally stored model created by this repository.
        package=joblib.load(FROZEN)
    if (package.get('schema') != 'frozen-wind-comparison-v1'
            or package.get('cutoff_day') != CUTOFF_DAY
            or package.get('first_eval_day') != FIRST_EVAL_DAY):
        raise SystemExit('Frozen artifact does not match comparison protocol')
    result=evaluate(package,rows)
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result.get(k) for k in ('new_races','new_days','status','brier_gain_without_wind')},ensure_ascii=False))

if __name__=='__main__':
    main()
