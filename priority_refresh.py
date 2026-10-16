"""BOAT12: deadline-first ordering and bounded performance instrumentation.

Usage: python priority_refresh.py [same arguments as conditions.py]
This wrapper does not change the models, probabilities, weather validity rules,
provider sources, number of downloads, or frequency of refreshes.
"""
from __future__ import annotations
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import conditions

JST = ZoneInfo('Asia/Tokyo')
STAGES = Path('history/conditions/latency/stages')


def iso_now():
    return datetime.now(JST).isoformat()


def run():
    # Historical model training remains exactly the original conditions.py CLI.
    if len(sys.argv) > 1:
        return conditions.main()
    started = time.monotonic()
    start_stamp = iso_now()
    events = []
    stats = {'generation_started_at': start_stamp, 'source_fetches': events}
    original_fetch = conditions.fetch_csv
    original_get_day = conditions.get_day
    original_estimate = conditions.condition_estimates
    counts = {'near': 0, 'evaluated': 0}

    def instrument_fetch(kind, day, session):
        begin = time.monotonic()
        when = iso_now()
        try:
            rows, info = original_fetch(kind, day, session)
        except Exception as exc:
            events.append({'kind': kind, 'started_at': when, 'elapsed_ms': round((time.monotonic()-begin)*1000), 'status': 'raised', 'error_type': type(exc).__name__})
            raise
        events.append({'kind': kind, 'started_at': when, 'elapsed_ms': round((time.monotonic()-begin)*1000), 'status': info.get('status'), 'rows': len(rows)})
        return rows, info

    def prioritized_day(day, historical, budget_end=None):
        recs, info = original_get_day(day, historical, budget_end)
        if not historical:
            now = datetime.now(JST)
            def key(row):
                cutoff = conditions.timestamp(row.get('cutoff'))
                # Prioritize unexpired races within 15 minutes, then other future races.
                # Closed races remain last; do not infer a deadline for missing dates.
                if cutoff is None or cutoff <= now:
                    return (2, float('inf'), row.get('code', ''))
                left = (cutoff-now).total_seconds()
                return (0 if left <= 15*60 else 1, left, row.get('code', ''))
            recs.sort(key=key)
            stats['fetch_and_join_finished_at'] = iso_now()
            stats['fetch_and_join_elapsed_ms'] = round((time.monotonic()-started)*1000)
            stats['near_deadline_first_15min'] = sum(0 < (conditions.timestamp(x.get('cutoff'))-now).total_seconds() <= 900 for x in recs if conditions.timestamp(x.get('cutoff')))
        return recs, info

    def instrument_estimate(record, package, now):
        cutoff = conditions.timestamp(record.get('cutoff'))
        left = (cutoff-now).total_seconds() if cutoff else None
        if left is not None and 0 < left <= 900:
            counts['near'] += 1
        counts['evaluated'] += 1
        return original_estimate(record, package, now)

    conditions.fetch_csv = instrument_fetch
    conditions.get_day = prioritized_day
    conditions.condition_estimates = instrument_estimate
    error = None
    try:
        # Original logic handles atomic conditions.json writes and any errors.
        conditions.update_today()
    except Exception as exc:
        error = exc
    finally:
        stats['generation_finished_at'] = iso_now()
        stats['generation_elapsed_ms'] = round((time.monotonic()-started)*1000)
        stats['races_evaluated'] = counts['evaluated']
        stats['races_within_15min'] = counts['near']
        stats['status'] = 'error' if error else 'ok'
        if error:
            stats['error_type'] = type(error).__name__
        try:
            STAGES.mkdir(parents=True, exist_ok=True)
            name = datetime.now(JST).strftime('%Y-%m-%d') + '.json'
            path = STAGES / name
            previous = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
            runs = previous.get('runs', []) if isinstance(previous, dict) else []
            if not isinstance(runs, list):
                runs = []
            runs.append(stats)
            # Bounded diagnostics prevent unbounded repository growth.
            payload = {'schema': 'boat12-stages-v1', 'date': name[:-5], 'runs': runs[-500:],
                       'notice': 'Measured inside GitHub Actions, not a proof of provider public availability or GitHub Pages publication time.'}
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            os.replace(temporary, path)
            print('PRIORITY TIMING:', stats['generation_elapsed_ms'], 'ms; near cutoff', counts['near'], '; status', stats['status'], flush=True)
        except Exception as exc:
            print('PRIORITY TIMING WRITE FAILED:', type(exc).__name__, str(exc)[:120], file=sys.stderr, flush=True)
    if error:
        raise error
    return 0


if __name__ == '__main__':
    raise SystemExit(run())
