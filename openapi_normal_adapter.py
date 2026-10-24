#!/usr/bin/env python3
"""BOAT12 normal-exhibition source compatibility report (research only).

Reads staged OpenAPI snapshots and the existing BOAT12 conditions feed.
Writes a separate JSON audit. Never edits conditions.json, probabilities,
the model or the published website, and never invents a weather observation time.
"""
import argparse
import json
import math
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

def timestamp(value):
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt if dt.tzinfo is not None else None
    except (TypeError, ValueError):
        return None

def number(value, low, high):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) and low <= result <= high else None
    except (ValueError, TypeError):
        return None

def compare(staged, conditions):
    result = {
        "schema": "boat12-openapi-normal-audit-v1",
        "date": staged.get("date"),
        "mode": "research_only_no_predictions",
        "counts": {},
        "records": {},
        "note": (
            "OpenAPI runner first-seen time is NOT weather observation time. "
            "This report never makes OpenAPI data eligible for existing model release."
        ),
    }
    if (staged.get("schema") != "boat12-openapi-precutoff-v1"
            or staged.get("date") != conditions.get("date")):
        result["error"] = "date_or_schema_mismatch"
        return result

    refs = {r.get("code"): r for r in conditions.get("races", []) if isinstance(r, dict)}
    counts = Counter()

    for code, snapshot in (staged.get("races") or {}).items():
        if not isinstance(snapshot, dict):
            continue
        counts["staged"] += 1
        ref = refs.get(code)
        reasons = []
        first_seen = timestamp(snapshot.get("observed_at"))
        cutoff = timestamp(snapshot.get("cutoff"))

        if ref is None:
            reasons.append("missing_boat12_reference")
        elif timestamp(ref.get("cutoff")) != cutoff:
            reasons.append("cutoff_mismatch")
        if not first_seen or not cutoff or first_seen >= cutoff - timedelta(seconds=30):
            reasons.append("not_safely_before_cutoff")
        if snapshot.get("registration_ids_verified") is not True:
            reasons.append("registration_ids_not_verified")

        players = snapshot.get("players")
        if not isinstance(players, list) or len(players) != 6:
            reasons.append("not_six_players")
        else:
            courses = []
            for i, player in enumerate(players, 1):
                if not isinstance(player, dict) or player.get("boat") != i:
                    reasons.append("boat_order_mismatch")
                    break
                if number(player.get("exhibition"), 4, 12) is None:
                    reasons.append("missing_exhibition")
                if number(player.get("exhibition_st"), -1, 2) is None:
                    reasons.append("missing_exhibition_st")
                course = number(player.get("course"), 1, 6)
                if course is None or not course.is_integer():
                    reasons.append("missing_course")
                else:
                    courses.append(int(course))
                if player.get("tilt") is not None and number(player["tilt"], -1, 4) is None:
                    reasons.append("invalid_tilt")
            if sorted(courses) != [1, 2, 3, 4, 5, 6]:
                reasons.append("invalid_course_order")

        if number(snapshot.get("wind_speed"), 0, 40) is None:
            reasons.append("missing_wind_speed")
        if number(snapshot.get("wave_cm"), 0, 200) is None:
            reasons.append("missing_wave_height")
        if snapshot.get("wind_direction_code") is None:
            reasons.append("missing_wind_direction")
        if snapshot.get("weather_code") is None:
            reasons.append("missing_weather_code")

        blockers = list(reasons)
        if not blockers:
            counts["normal_basic_fields_valid"] += 1

        # A snapshot's observed_at records the GitHub runner receiving it,
        # not the original meteorological instrument's observation time.
        reasons.append("weather_observation_time_unverified")
        counts["weather_time_unverified"] += 1

        result["records"][code] = {
            "venue": ref.get("venue") if ref else None,
            "race": ref.get("race") if ref else None,
            "received_at": snapshot.get("observed_at"),
            "cutoff": snapshot.get("cutoff"),
            "normal_basic_fields_valid": not blockers,
            "normal_model_release_eligible": False,
            "reasons": sorted(set(reasons)),
        }
    result["counts"] = dict(counts)
    return result

def self_test():
    day = "2026-10-24"
    code = "202610240508"
    cutoff = "2026-10-24T13:53:00+09:00"
    racers = [{"boat": i, "exhibition": 6.7, "exhibition_st": .12,
               "course": i, "tilt": 0} for i in range(1, 7)]
    staged = {"schema": "boat12-openapi-precutoff-v1", "date": day, "races": {
        code: {"observed_at": "2026-10-24T13:44:45+09:00", "cutoff": cutoff,
               "registration_ids_verified": True, "players": racers,
               "wind_speed": 1, "wave_cm": 2, "wind_direction_code": 5,
               "weather_code": 3}}}
    conditions = {"date": day, "races": [{"code": code, "cutoff": cutoff,
                    "venue": "多摩川", "race": 8}]}
    result = compare(staged, conditions)
    assert result["counts"]["normal_basic_fields_valid"] == 1
    assert not result["records"][code]["normal_model_release_eligible"]
    staged["races"][code]["observed_at"] = "2026-10-24T13:54:00+09:00"
    assert "not_safely_before_cutoff" in compare(staged, conditions)["records"][code]["reasons"]
    print("SELF TEST PASSED")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staged", default="openapi-precutoff.json")
    parser.add_argument("--conditions", default="conditions.json")
    parser.add_argument("--out", default="openapi-normal-audit.json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    with open(args.staged, encoding="utf-8") as handle:
        staged = json.load(handle)
    with open(args.conditions, encoding="utf-8") as handle:
        conditions = json.load(handle)
    result = compare(staged, conditions)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["counts"], ensure_ascii=False))

if __name__ == "__main__":
    main()
