#!/usr/bin/env python3
"""Separate research-only BOAT12 basic pre-deadline trifecta model.

Uses audited pre-deadline dataset and same time-split/matched baseline as
trifecta_live_train.py. Never writes published trifecta_model.json.
"""
import argparse
from pathlib import Path
import numpy as np
import trifecta_live_train as research

REPORT_PATH = Path('trifecta_basic_model_report.json')
MODEL_PATH = Path('history/trifecta/basic_model_research.json')


def basic_live_features(rows):
    """30-column stage residual inputs; only 0:9 and 13:29 are populated.

    The shared research code is reused so training, calibration and chronological
    comparison methodology remain identical. Original exhibition, including
    lap/half-lap, is deliberately NOT used. An absent original is never imputed.
    """
    x = np.zeros((len(rows), 6, 30), dtype=float)
    for n, row in enumerate(rows):
        f = row['features']
        wind = float(f['wind_speed']) / 10
        wave = float(f['wave_cm']) / 20
        direction = int(f['wind_direction_code'])
        if direction not in range(17):
            raise ValueError('invalid wind direction')
        for i in range(1, 7):
            b = i - 1
            c = float(f[f'b{i}_exhibition_course'])
            if not 1 <= c <= 6 or int(c) != c:
                raise ValueError('invalid exhibition course')
            z = x[n, b]
            z[0] = float(f[f'b{i}_exhibition'])
            z[1] = float(f[f'b{i}_exhibition_st'])
            z[2] = (c - i) / 5
            z[3] = (c - 3.5) / 3
            z[4] = float(f[f'b{i}_tilt'])
            z[5] = wind * (i - 3.5) / 3
            z[6] = wave * (i - 3.5) / 3
            z[7] = wind * (c - 3.5) / 3
            z[8] = wave * (c - 3.5) / 3
            z[13 + direction] = (i - 3.5) / 3
        for j, scale in ((0, .35), (1, .15), (4, 1)):
            x[n, :, j] = (x[n, :, j] - x[n, :, j].mean()) / scale
    if not np.isfinite(x).all():
        raise ValueError('nonfinite feature values')
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        sample = {'features': {'wind_speed': 3, 'wave_cm': 2, 'wind_direction_code': 4}}
        for i in range(1, 7):
            sample['features'].update({
                f'b{i}_exhibition': 6.6 + i * .05,
                f'b{i}_exhibition_st': .1,
                f'b{i}_exhibition_course': i,
                f'b{i}_tilt': -.5,
            })
        out = basic_live_features([sample])
        assert out.shape == (1, 6, 30) and np.isfinite(out).all()
        assert np.all(out[:, :, 9:13] == 0)
        assert np.count_nonzero(out[:, :, 13:]) > 0
        print('basic live feature tests: PASS')
        return
    # Explicitly reuse the existing audited-data fit, which saves a research-only
    # artifact; redirect its paths so the existing full live research remains intact.
    research.live_features = basic_live_features
    research.REPORT = REPORT_PATH
    research.MODEL = MODEL_PATH
    report = research.run()
    # Note: research.run() writes publication=false / ready_for_publication=false.
    print('BASIC_RESULT', report['status'], 'ROWS', report['rows'], flush=True)


if __name__ == '__main__':
    main()
