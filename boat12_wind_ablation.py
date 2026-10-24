#!/usr/bin/env python3
"""Read-only A/B retraining: normal model vs normal without wind features.

Writes ONLY an experiment report. No modification to production model or gates.
"""
import argparse
import json
from collections import Counter
from datetime import date, timedelta, datetime, timezone
from pathlib import Path

WIND_FIELDS = {"wind_speed", "wind_direction", "venue_wind_direction"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--out", default="history/conditions/wind_ablation.json")
    args = parser.parse_args()
    import conditions
    if args.self_test:
        sample = {"wind_speed": 2, "wind_direction": "3", "venue_wind_direction": "01:3", "wave_cm": 4}
        assert set(sample) - WIND_FIELDS == {"wave_cm"}
        assert conditions.features and conditions.fit_model and conditions.predict
        print("SELF TEST PASSED")
        return

    import numpy as np
    from sklearn.metrics import brier_score_loss, log_loss
    rows, seen = [], set()
    root = conditions.ROOT
    report_path = root / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    sr = report["scopes"]["normal"]
    if not sr.get("test_start"):
        raise SystemExit("No fixed model holdout split in report")
    # Read only the EXACT data period belonging to the existing report.
    start, cal_start, test_start, end = sr["period_start"], sr["calibration_start"], sr["test_start"], sr["period_end"]
    for file in sorted((root / "days").glob("*.jsonl.gz")):
        for row in conditions.read_records(file):
            if (row.get("version") == conditions.VERSION and row.get("basic_ready") is True
                    and row.get("label") in (0, 1) and start <= row.get("date", "") <= end
                    and row.get("code") not in seen):
                rows.append(row)
                seen.add(row["code"])
    rows.sort(key=lambda x: x["code"])
    train = [r for r in rows if r["date"] < cal_start]
    cal = [r for r in rows if cal_start <= r["date"] < test_start]
    test = [r for r in rows if r["date"] >= test_start]
    expected = [sr.get("train_rows"), sr.get("calibration_rows"), sr.get("test_rows")]
    actual = list(map(len, (train, cal, test)))
    if actual != expected:
        raise SystemExit(f"Split row counts differ; expected {expected}, got {actual}. Report not written.")
    original_features = conditions.features
    try:
        full = conditions.fit_model(train, cal, "normal")
        full_prob = conditions.predict(full, test)

        def no_wind_features(record, scope):
            features, categories = original_features(record, scope)
            return ({k: v for k, v in features.items() if k not in WIND_FIELDS},
                    [c for c in categories if c not in WIND_FIELDS])

        conditions.features = no_wind_features
        no_wind = conditions.fit_model(train, cal, "normal")
        no_wind_prob = conditions.predict(no_wind, test)
    finally:
        conditions.features = original_features

    y = np.array([r["label"] for r in test])
    brier_full = float(brier_score_loss(y, full_prob))
    brier_no_wind = float(brier_score_loss(y, no_wind_prob))
    ci = conditions.day_bootstrap_gain([r["date"] for r in test],
                  (full_prob - y) ** 2 - (no_wind_prob - y) ** 2)
    result = {
        "schema": "boat12-normal-wind-ablation-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_report_generated_at": report.get("generated_at"),
        "split": {"train": len(train), "calibration": len(cal), "test": len(test),
                  "calibration_start": cal_start, "test_start": test_start, "test_end": end},
        "removed_features": sorted(WIND_FIELDS),
        "full": {"brier": brier_full, "logloss": float(log_loss(y, full_prob))},
        "without_wind": {"brier": brier_no_wind, "logloss": float(log_loss(y, no_wind_prob))},
        "brier_gain_without_wind": brier_full - brier_no_wind,
        "day_bootstrap_95ci": ci,
        "conclusion": "promising_needs_fresh_holdout" if ci[0] > 0 and brier_no_wind < brier_full else "not_proven",
        "production_changed": False,
        "warning": "This holdout was already used in model development. This experiment is exploratory; future untouched races are needed before release. No win/profit claims.",
    }
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"full_brier": brier_full, "without_wind_brier": brier_no_wind,
                      "gain": brier_full - brier_no_wind, "ci95": ci}, ensure_ascii=False))


if __name__ == "__main__":
    main()
