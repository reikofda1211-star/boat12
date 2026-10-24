#!/usr/bin/env python3
"""BOAT12 read-only normal-model feature importance diagnostic.

Permutes whole, within-race feature groups across held-out races. This is NOT
retraining and is NOT causal proof that a feature improves forecasting.
Never changes the production model, release criteria or conditions.json.
"""
from __future__ import annotations
import argparse
import copy
import json
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

GROUPS = {
    "wind": ("weather", ("wind_speed", "wind_direction_code")),
    "waves": ("weather", ("wave_cm",)),
    "exhibition_time": ("players", ("exhibition",)),
    "exhibition_st": ("players", ("exhibition_st",)),
    "exhibition_course": ("players", ("course",)),
    "tilt": ("players", ("tilt",)),
}


def permuted(rows, field_group, rng):
    """Swap a complete race's field group; preserve six-boat structure."""
    mode, keys = GROUPS[field_group]
    indexes = rng.permutation(len(rows))
    result = [dict(r) for r in rows]
    for i, j in enumerate(indexes):
        if mode == "weather":
            result[i]["weather"] = dict(rows[i]["weather"])
            for key in keys:
                result[i]["weather"][key] = rows[int(j)]["weather"].get(key)
        else:
            result[i]["players"] = [dict(p) for p in rows[i]["players"]]
            for boat in range(6):
                for key in keys:
                    result[i]["players"][boat][key] = rows[int(j)]["players"][boat].get(key)
    return result


def audit(rows, model, repeats=3, seed=1212):
    import numpy as np
    from sklearn.metrics import brier_score_loss, log_loss
    import conditions
    labels = np.asarray([r["label"] for r in rows], dtype=int)
    original = conditions.predict(model, rows)
    baseline = float(brier_score_loss(labels, original))
    output = {"baseline_brier": baseline, "baseline_logloss": float(log_loss(labels, original)),
              "rows": len(rows), "permutations_per_group": repeats, "groups": {}}
    rng = np.random.default_rng(seed)
    for group in GROUPS:
        scores = []
        for _ in range(repeats):
            shuffled = permuted(rows, group, rng)
            probabilities = conditions.predict(model, shuffled)
            scores.append(float(brier_score_loss(labels, probabilities)))
        mean = float(np.mean(scores))
        output["groups"][group] = {"shuffled_brier_mean": mean,
                                   "delta_brier": mean-baseline,
                                   "shuffled_brier_each": scores}
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="history/conditions/normal_feature_audit.json")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        import numpy as np
        example = [{"weather": {"wind_speed": k, "wave_cm": 1},
                    "players": [{"course": j, "exhibition": k} for j in range(1,7)]} for k in range(8)]
        shuffled = permuted(example, "exhibition_course", np.random.default_rng(5))
        assert all(sorted(p["course"] for p in race["players"]) == list(range(1,7)) for race in shuffled)
        assert all(shuffled[i]["weather"] == example[i]["weather"] for i in range(8))
        print("SELF TEST PASSED")
        return
    if not 1 <= args.repeats <= 10:
        ap.error("repeats must be between 1 and 10")
    import joblib
    import conditions
    model_path = Path("history/conditions/model.joblib")
    if not model_path.exists():
        raise SystemExit("Model artifact not found; no analysis performed")
    package = joblib.load(model_path)  # own trusted repository model only
    report = package.get("report", {})
    normal = package.get("scopes", {}).get("normal")
    scope_report = report.get("scopes", {}).get("normal", {})
    if not normal or not normal.get("conditions") or not scope_report.get("test_start"):
        raise SystemExit("Normal model or holdout definition absent")
    start = scope_report["test_start"]
    end = scope_report["period_end"]
    rows, seen = [], set()
    for file in sorted(Path("history/conditions/days").glob("*.jsonl.gz")):
        for row in conditions.read_records(file):
            if (row.get("date") and start <= row["date"] <= end
                    and row.get("basic_ready") is True and row.get("label") in (0,1)
                    and row.get("code") not in seen):
                rows.append(row)
                seen.add(row["code"])
    rows.sort(key=lambda r: r["code"])
    if len(rows) != scope_report.get("test_rows"):
        raise SystemExit(f"Holdout size mismatch: current {len(rows)} vs trained {scope_report.get('test_rows')}; refusing to compare")
    result = audit(rows, normal["conditions"], args.repeats)
    result.update({"schema": "boat12-normal-permutation-audit-v1",
                   "created_at": datetime.now(timezone.utc).isoformat(),
                   "test_start": start, "test_end": end,
                   "model_report_generated_at": report.get("generated_at"),
                   "release_rule_unchanged": True,
                   "note": "Read-only permutation sensitivity on the original held-out set. Positive delta indicates the fitted model used that feature group on these races, NOT a proven accuracy improvement from training with it. This holdout has already influenced model selection; a fresh prospective validation is needed."})
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"rows":len(rows),"baseline_brier":result["baseline_brier"],
                      "delta_brier":{k:round(v["delta_brier"],6) for k,v in result["groups"].items()}},ensure_ascii=False))

if __name__ == "__main__":
    main()
