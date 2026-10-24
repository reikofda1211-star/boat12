"""Walk-forward evaluation of BOAT12 research models; does not modify production."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from compare_two_stage import load, pipeline
from compare_competitor_features import competitor_features, evaluate
from model_v2 import CATEGORICAL, NUMERICAL


def main():
    rows = load()
    for _, features, _, _ in rows:
        features.update(competitor_features(features))
    dates = sorted(set(r[0] for r in rows))
    # Five expanding-window folds, with at least 50% of dates for initial training.
    boundaries = np.linspace(len(dates)//2, len(dates), 6, dtype=int)
    reports = []
    for i in range(5):
        train = [r for r in rows if r[0] < dates[boundaries[i]]]
        test = [r for r in rows if dates[boundaries[i]] <= r[0] < (dates[boundaries[i+1]] if boundaries[i+1] < len(dates) else '9999')]
        if not train or not test:
            continue
        xtr = pd.DataFrame([r[1] for r in train], columns=CATEGORICAL+NUMERICAL)
        xte = pd.DataFrame([r[1] for r in test], columns=CATEGORICAL+NUMERICAL)
        ytr = np.array([r[3] for r in train]); y = np.array([r[3] for r in test])
        direct = pipeline().fit(xtr, ytr).predict_proba(xte)[:,1]
        base = evaluate(train, test, False)
        enhanced = evaluate(train, test, True)
        scores = {}
        for name, prob in [('direct',direct),('two_stage',base),('competitor',enhanced)]:
            scores[name] = {'brier':float(brier_score_loss(y,prob)), 'logloss':float(log_loss(y,prob)), 'auc':float(roc_auc_score(y,prob)) if len(set(y))==2 else None}
        reports.append({'fold':i+1,'train_rows':len(train),'test_rows':len(test),'test_start':test[0][0],'test_end':test[-1][0], 'scores':scores})
    print(json.dumps({'method':'expanding window, 5 non-overlapping test periods','folds':reports,'note':'Research only; no production changes. Compare fold-by-fold, not just the average.'},ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
