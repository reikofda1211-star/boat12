"""Chronological offline comparison of direct and two-stage 1-2 forecasts.

The production prediction model is untouched. Labels for first/second place
come from history/races.csv; card features come from history/training.csv.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model_v2 import CATEGORICAL, NUMERICAL, feature_row

CARDS = Path('history/training.csv')
RESULTS = Path('history/races.csv')
OUTPUT = Path('history/two_stage_comparison.json')


def load():
    with RESULTS.open(encoding='utf-8', newline='') as f:
        results = {(r['date'], r['venue'], r['race']): r['result12'] for r in csv.DictReader(f)}
    seen = set()
    records = []
    with CARDS.open(encoding='utf-8', newline='') as f:
        for row in csv.DictReader(f):
            key = (row['date'], row['venue'], row['race'])
            outcome = results.get(key, '')
            if key in seen or len(outcome) != 3 or outcome[1] != '-' or outcome[0] not in '123456' or outcome[2] not in '123456' or outcome[0] == outcome[2]:
                continue
            seen.add(key)
            records.append((row['date'], feature_row(row), int(outcome[0] == '1'), int(outcome == '1-2')))
    return sorted(records, key=lambda r: r[0])


def pipeline():
    transform = ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True)), ('scale', StandardScaler())]), NUMERICAL),
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
    ])
    return Pipeline([('features', transform), ('classifier', LogisticRegression(C=0.03, max_iter=2000))])


def scores(actual, prob):
    return {'brier': round(float(brier_score_loss(actual, prob)), 6),
            'logloss': round(float(log_loss(actual, prob)), 6),
            'auc': round(float(roc_auc_score(actual, prob)), 5)}


def main():
    rows = load()
    days = sorted({r[0] for r in rows})
    if len(rows) < 1200 or len(days) < 30:
        raise SystemExit('Insufficient history for a chronological holdout')
    split = days[int(0.8 * len(days))]
    train = [r for r in rows if r[0] < split]
    test = [r for r in rows if r[0] >= split]
    x_train = pd.DataFrame([r[1] for r in train], columns=CATEGORICAL + NUMERICAL)
    x_test = pd.DataFrame([r[1] for r in test], columns=CATEGORICAL + NUMERICAL)
    y_first = np.array([r[2] for r in train])
    y_direct = np.array([r[3] for r in train])
    y_test = np.array([r[3] for r in test])
    first = pipeline().fit(x_train, y_first)
    conditioned = pipeline().fit(x_train.loc[y_first == 1], y_direct[y_first == 1])
    direct = pipeline().fit(x_train, y_direct)
    p_first = first.predict_proba(x_test)[:, 1]
    p_second_given_first = conditioned.predict_proba(x_test)[:, 1]
    p_two_stage = p_first * p_second_given_first
    p_direct = direct.predict_proba(x_test)[:, 1]
    gain = (p_direct - y_test) ** 2 - (p_two_stage - y_test) ** 2
    by_day = {}
    for item, record in zip(gain, test):
        total, count = by_day.get(record[0], (0.0, 0))
        by_day[record[0]] = (total + float(item), count + 1)
    day_values = np.array(list(by_day.values()))
    rng = np.random.default_rng(617)
    sample = rng.integers(0, len(day_values), size=(1500, len(day_values)))
    boot = day_values[sample]
    means = boot[:, :, 0].sum(axis=1) / boot[:, :, 1].sum(axis=1)
    ci = [round(float(n), 6) for n in np.quantile(means, [0.025, 0.975])]
    out = {'train_rows': len(train), 'test_rows': len(test), 'test_start': split,
           'direct': scores(y_test, p_direct), 'two_stage': scores(y_test, p_two_stage),
           'delta_brier_two_stage_minus_direct': round(float(brier_score_loss(y_test, p_two_stage) - brier_score_loss(y_test, p_direct)), 6),
           'brier_gain_day_bootstrap_95ci': ci,
           'test_actual_12_rate': round(float(y_test.mean()), 5),
           'note': 'Offline experiment; same chronological holdout, no production deployment.'}
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
