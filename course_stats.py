"""Build leakage-safe, pre-race course statistics for BOAT12.

Usage: python course_stats.py --start 2026-01-01 --end 2026-10-25

Input: BoatraceCSV's published race cards and realtime results.  A row's
statistics use results from *earlier calendar days only*. No same-day
outcomes or future races are included. The output is experimental and does
not change production probabilities.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import time
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


ROOT = "https://boatracecsv.github.io/data"
PSEUDO_STARTS = 20  # Regularize small course samples toward the prior.
OUTPUT_FIELDS = ["date", "code", "boat", "registration", "exhibition_course",
                 "history_starts", "history_wins", "history_seconds",
                 "course_win_rate", "course_second_rate", "prior_win_rate",
                 "prior_second_rate", "source_days"]


def normalize(value):
    return str(value or "").strip()


def get_rows(kind: str, day: date, retries: int = 3):
    name = f"{ROOT}/{kind}/{day:%Y/%m/%d}.csv"
    for attempt in range(retries):
        try:
            with urlopen(name, timeout=25) as response:
                data = response.read().decode("utf-8-sig")
            return list(csv.DictReader(io.StringIO(data)))
        except HTTPError as exc:
            if exc.code == 404:
                return []
            if exc.code not in (429, 500, 502, 503, 504):
                raise
        except (TimeoutError, URLError):
            pass
        if attempt < retries - 1:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Could not fetch {name}; aborting to avoid incomplete stats")


def rows_by_code(rows):
    return {normalize(r.get("レースコード")): r for r in rows
            if re.fullmatch(r"\d{12}", normalize(r.get("レースコード")))}


def sample_stat(counts, course):
    starts, wins, seconds = counts[course]
    return starts, wins, seconds


def build(start: date, end: date, output: Path, dry_run: bool = False):
    if end >= date.today():
        raise ValueError("End must be before today; partial current-day data are not allowed")
    if start > end:
        raise ValueError("Start is later than end")
    # (registration, actual course) -> [starts, firsts, seconds]
    athlete = defaultdict(lambda: [0, 0, 0])
    # Actual course aggregate used only as a prior, not as a future-data estimate.
    pooled = defaultdict(lambda: [0, 0, 0])
    out = []
    seen_days = 0
    day = start
    while day <= end:
        cards = rows_by_code(get_rows("programs/race_cards", day))
        results = rows_by_code(get_rows("results/realtime", day))
        stt = rows_by_code(get_rows("previews/stt", day))
        pending = []
        # Snapshot each day's features before applying ANY results from that day.
        for code, card in cards.items():
            race_result = results.get(code)
            race_stt = stt.get(code)
            if not race_result or not race_stt:
                continue
            registration = [normalize(card.get(f"艇{i}_登録番号")) for i in range(1, 7)]
            if len(set(registration)) != 6 or not all(re.fullmatch(r"\d{4}", n) for n in registration):
                continue
            actual_course = {}
            for course in range(1, 7):
                boat = normalize(race_result.get(f"{course}コース_艇番"))
                if boat.isdigit() and 1 <= int(boat) <= 6:
                    actual_course[int(boat)] = course
            if len(actual_course) != 6 or len(set(actual_course.values())) != 6:
                continue
            finish = {}
            for rank in (1, 2):
                boat = normalize(race_result.get(f"{rank}着_艇番"))
                if boat.isdigit() and 1 <= int(boat) <= 6:
                    finish[int(boat)] = rank
            if sorted(finish.values()) != [1, 2]:
                continue
            predicted = {}
            for boat in range(1, 7):
                c = normalize(race_stt.get(f"艇{boat}_コース"))
                if not c.isdigit() or not 1 <= int(c) <= 6:
                    break
                predicted[boat] = int(c)
            if len(predicted) != 6 or len(set(predicted.values())) != 6:
                continue
            for boat in range(1, 7):
                reg, course = registration[boat - 1], predicted[boat]
                starts, wins, seconds = athlete[(reg, course)]
                pool_n, pool_w, pool_s = pooled[course]
                prior_w = pool_w / pool_n if pool_n else 1 / 6
                prior_s = pool_s / pool_n if pool_n else 1 / 6
                row = {"date": day.isoformat(), "code": code, "boat": boat,
                       "registration": reg, "exhibition_course": course,
                       "history_starts": starts, "history_wins": wins,
                       "history_seconds": seconds,
                       "course_win_rate": round((wins + PSEUDO_STARTS * prior_w) / (starts + PSEUDO_STARTS), 6),
                       "course_second_rate": round((seconds + PSEUDO_STARTS * prior_s) / (starts + PSEUDO_STARTS), 6),
                       "prior_win_rate": round(prior_w, 6),
                       "prior_second_rate": round(prior_s, 6), "source_days": seen_days}
                out.append(row)
            # Late-published results must never be used for a subsequent day's
            # supposedly pre-race statistics. Omit those observations from
            # the rolling counters instead of peeking into future data.
            result_available = normalize(race_result.get("取得日時"))
            if result_available.startswith(day.isoformat()):
                pending.append((registration, actual_course, finish))
        for regs, actual, finish in pending:
            for boat, course in actual.items():
                wins = int(finish.get(boat) == 1)
                seconds = int(finish.get(boat) == 2)
                for arr in (athlete[(regs[boat - 1], course)], pooled[course]):
                    arr[0] += 1
                    arr[1] += wins
                    arr[2] += seconds
        seen_days += 1
        day += timedelta(days=1)
    if not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_suffix(output.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows(out)
        tmp.replace(output)
    print(json.dumps({"records": len(out), "days": seen_days, "output": None if dry_run else str(output),
                      "example": out[-1] if out else None}, ensure_ascii=False))


def self_test():
    counts = defaultdict(lambda: [0, 0, 0])
    assert sample_stat(counts, 1) == (0, 0, 0)
    counts[1] = [10, 5, 2]
    assert sample_stat(counts, 1) == (10, 5, 2)
    assert abs((5 + 20 / 6) / 30 - 0.2777777777777778) < 1e-9
    print("SELF TEST PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat)
    parser.add_argument("--output", type=Path, default=Path("history/course_stats.csv"))
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if args.start is None or args.end is None:
            parser.error("--start and --end are required")
        build(args.start, args.end, args.output, args.dry_run)
