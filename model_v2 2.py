"""Boat 1-2 experimental validation v2.

Run after model_pipeline.py has created history/training.csv.
Never represent these probabilities as validated until the held-out Brier score
beats a constant-rate baseline. This does not modify the live index.html.
"""
import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

INPUT = Path('history/training.csv')
OUTPUT = Path('history/model_v2_report.json')
MODEL = Path('history/model_v2.joblib')
VENUES = [
    '桐生', '戸田', '江戸川', '平和島', '多摩川', '浜名湖', '蒲郡',
    '常滑', '津', '三国', 'びわこ', '住之江', '尼崎', '鳴門',
    '丸亀', '児島', '宮島', '徳山', '下関', '若松', '芦屋',
    '福岡', '唐津', '大村'
]
METRICS = ['win', 'rate', 'local_win', 'local_rate', 'motor_rate', 'boat_rate']
CATEGORICAL = ['venue', 'race']
NUMERICAL = [f'b{i}_{m}' for i in range(1, 7) for m in METRICS]
NUMERICAL += [f'diff12_{m}' for m in METRICS]
NUMERICAL += [f'diff1_others_{m}' for m in METRICS]
NUMERICAL += [f'diff2_others_{m}' for m in METRICS]


def num(raw, minimum, maximum):
    try:
        n = float(raw)
        return n if math.isfinite(n) and minimum <= n <= maximum else np.nan
    except (ValueError, TypeError):
        return np.nan


def limits(metric):
    if metric.endswith('rate'):
        return (0, 100)
    return (0, 10)


def feature_row(r):
    out = {'venue': r['venue'], 'race': str(int(r['race']))}
    for m in METRICS:
        lo, hi = limits(m)
        vals = [num(r.get(f'b{i}_{m}'), lo, hi) for i in range(1, 7)]
        for i, v in enumerate(vals, start=1):
            out[f'b{i}_{m}'] = v
        out[f'diff12_{m}'] = vals[0] - vals[1]
        rest = [v for v in vals[2:] if not np.isnan(v)]
        mean_rest = float(np.mean(rest)) if rest else np.nan
        out[f'diff1_others_{m}'] = vals[0] - mean_rest
        out[f'diff2_others_{m}'] = vals[1] - mean_rest
    return out


def main():
    if not INPUT.exists():
        raise SystemExit('history/training.csv がありません。先に既存の学習ワークフローを実行してください。')
    with INPUT.open(encoding='utf-8', newline='') as f:
        raw = list(csv.DictReader(f))

    required = {'date', 'venue', 'race', 'hit12'} | {
        f'b{i}_{m}' for i in range(1, 7) for m in METRICS
    }
    if not raw or not required.issubset(set(raw[0])):
        raise SystemExit('学習CSVに必要な列がありません。列名と番組表の解析を確認してください。')

    unique = {}
    excluded = Counter()
    for r in raw:
        try:
            date = datetime.strptime(r['date'], '%Y%m%d')
            race = int(r['race'])
            if r['venue'] not in VENUES or not 1 <= race <= 12:
                excluded['bad_venue_or_race'] += 1
                continue
            if r['hit12'] not in ('0', '1'):
                excluded['bad_label'] += 1
                continue
            key = (date.strftime('%Y%m%d'), r['venue'], race)
            if key in unique:
                excluded['duplicate'] += 1
                continue
            unique[key] = r
        except (ValueError, KeyError, TypeError):
            excluded['bad_date'] += 1

    rows = [unique[k] for k in sorted(unique)]
    dates = sorted({r['date'] for r in rows})
    report = {
        'generated_at_jst': datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
        'source_rows': len(raw), 'valid_rows': len(rows),
        'days': len(dates), 'excluded': dict(excluded),
        'status': 'not_enough_data',
        'warning': '実験的な検証。的中・利益・確率の正確さを保証しません。'
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    if len(rows) < 1200 or len(dates) < 30:
        OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print('学習保留: レース1200件以上、30日以上が必要')
        return

    # Chronological 80/20 split: no future race is used for training.
    split_day = dates[int(len(dates) * 0.8)]
    train = [r for r in rows if r['date'] < split_day]
    valid = [r for r in rows if r['date'] >= split_day]
    y_train = np.array([int(r['hit12']) for r in train])
    y_valid = np.array([int(r['hit12']) for r in valid])
    if len(train) < 600 or len(valid) < 200 or len(set(y_train)) < 2 or len(set(y_valid)) < 2:
        print('学習保留: 訓練/検証数または正例が不足')
        OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        return

    def matrix(rows_):
        processed = [feature_row(r) for r in rows_]
        # Pandas is already in requirements.txt.
        import pandas as pd
        return pd.DataFrame(processed, columns=CATEGORICAL + NUMERICAL)

    transformer = ColumnTransformer([
        ('numeric', Pipeline([
            ('impute', SimpleImputer(strategy='median', add_indicator=True)),
            ('scale', StandardScaler())
        ]), NUMERICAL),
        ('category', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL)
    ])
    model = Pipeline([
        ('features', transformer),
        ('classifier', LogisticRegression(C=0.03, max_iter=2000))
    ])
    model.fit(matrix(train), y_train)
    probability = model.predict_proba(matrix(valid))[:, 1]
    base_rate = float(y_train.mean())
    brier_model = float(brier_score_loss(y_valid, probability))
    brier_baseline = float(brier_score_loss(y_valid, np.full(len(y_valid), base_rate)))
    improved = brier_model < brier_baseline
    report.update({
        'status': 'validated_better_than_constant' if improved else 'not_improved',
        'training_rows': len(train), 'validation_rows': len(valid),
        'validation_start': split_day, 'training_hit_rate': round(base_rate, 5),
        'validation_hit_rate': round(float(y_valid.mean()), 5),
        'brier_model': round(brier_model, 5),
        'brier_baseline': round(brier_baseline, 5),
        'auc': round(float(roc_auc_score(y_valid, probability)), 5),
        'logloss': round(float(log_loss(y_valid, probability)), 5),
        'improved_over_baseline': bool(improved),
        'next_step': '出走表の列対応、取得漏れ、年度差、較正をさらに検証する。'
    })
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    # Retain only experimental model; do not overwrite live site.
    joblib.dump({'pipeline': model, 'columns': CATEGORICAL + NUMERICAL,
                 'metrics': METRICS, 'report': report}, MODEL)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
