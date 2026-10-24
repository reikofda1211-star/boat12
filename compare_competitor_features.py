"""Research-only comparison: two-stage 1-2 with 3-6 competitor strength features.

Run: python compare_competitor_features.py
No live model, probability, workflow or published file is changed.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from compare_two_stage import load
from model_v2 import CATEGORICAL, NUMERICAL, METRICS


def competitor_features(row):
    result = {}
    for metric in METRICS:
        second = row[f'b2_{metric}']
        others = np.array([row[f'b{i}_{metric}'] for i in range(3, 7)], dtype=float)
        good = others[np.isfinite(others)]
        result[f'comp_max_{metric}'] = float(good.max()) if len(good) else np.nan
        result[f'comp_top2_{metric}'] = float(np.mean(np.sort(good)[-2:])) if len(good) >= 2 else np.nan
        result[f'comp_gap2_{metric}'] = second - float(good.max()) if len(good) and np.isfinite(second) else np.nan
        result[f'comp_above2_{metric}'] = float((good > second).sum()) if len(good) == 4 and np.isfinite(second) else np.nan
    return result


EXTRA = [f'{name}_{m}' for m in METRICS for name in ('comp_max', 'comp_top2', 'comp_gap2', 'comp_above2')]


def make_pipeline(numeric):
    transform = ColumnTransformer([
        ('numeric', Pipeline([('fill', SimpleImputer(strategy='median', add_indicator=True)), ('scale', StandardScaler())]), numeric),
        ('categorical', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
    ])
    return Pipeline([('features', transform), ('classifier', LogisticRegression(C=0.03, max_iter=2000))])


def evaluate(train, test, enriched):
    numeric = NUMERICAL + (EXTRA if enriched else [])
    columns = CATEGORICAL + numeric
    x_train = pd.DataFrame([r[1] for r in train], columns=columns)
    x_test = pd.DataFrame([r[1] for r in test], columns=columns)
    first_train = np.array([r[2] for r in train])
    y_train = np.array([r[3] for r in train])
    first = make_pipeline(numeric).fit(x_train, first_train)
    second = make_pipeline(numeric).fit(x_train.loc[first_train == 1], y_train[first_train == 1])
    return first.predict_proba(x_test)[:, 1] * second.predict_proba(x_test)[:, 1]


def main():
    rows = load()
    for _, row, _, _ in rows:
        row.update(competitor_features(row))
    days = sorted({r[0] for r in rows})
    split = days[int(len(days) * 0.8)]
    train = [r for r in rows if r[0] < split]
    test = [r for r in rows if r[0] >= split]
    y = np.array([r[3] for r in test])
    base = evaluate(train, test, False)
    new = evaluate(train, test, True)
    gain = (base - y)**2 - (new - y)**2
    days_gain = defaultdict(lambda: [0.0, 0])
    for value, row in zip(gain, test):
        days_gain[row[0]][0] += float(value)
        days_gain[row[0]][1] += 1
    blocks = np.array(list(days_gain.values()))
    rng = np.random.default_rng(618)
    indexes = rng.integers(0, len(blocks), size=(1500, len(blocks)))
    samples = blocks[indexes]
    gain_ci = np.quantile(samples[:, :, 0].sum(axis=1) / samples[:, :, 1].sum(axis=1), [0.025, 0.975])
    result = {
        'train_rows': len(train), 'test_rows': len(test), 'test_start': split,
        'baseline_brier': float(brier_score_loss(y, base)),
        'competitor_brier': float(brier_score_loss(y, new)),
        'baseline_logloss': float(log_loss(y, base)),
        'competitor_logloss': float(log_loss(y, new)),
        'baseline_auc': float(roc_auc_score(y, base)),
        'competitor_auc': float(roc_auc_score(y, new)),
        'brier_gain_bootstrap_95ci': gain_ci.tolist(),
        'warning': 'Research-only. A positive result on one holdout is not sufficient for production adoption.'
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
