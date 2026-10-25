#!/usr/bin/env python3
"""Read-only BOAT12 availability/latency diagnostic. No prediction changes."""
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

CONDITIONS = Path('conditions.json')
OPENAPI = Path('openapi-status.json')
STAGED = Path('openapi-precutoff.json')
LATENCY_ROOT = Path('history/conditions/latency')
OUTPUT = Path('history/conditions/refresh_diagnostic.json')

def load(path):
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}

def parse(value):
    try:
        dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return dt if dt.tzinfo else None
    except (TypeError, ValueError):
        return None

def gap(cutoff, stamp):
    c, s = parse(cutoff), parse(stamp)
    return round((c-s).total_seconds()) if c and s else None

def main():
    conditions = load(CONDITIONS)
    openapi = load(OPENAPI)
    staged = load(STAGED)
    day = conditions.get('date')
    if not day:
        raise SystemExit('No conditions.json with date')
    races = conditions.get('races') or []
    api_races = {}
    if openapi.get('date') == day:
        for venue in (openapi.get('venues') or {}).values():
            for race in (venue.get('races') or []):
                api_races[race.get('code')] = race
    staged_races = staged.get('races', {}) if staged.get('date') == day else {}
    latency = load(LATENCY_ROOT / (day + '.json'))
    seen_races = latency.get('races', {}) if latency.get('date') == day else {}
    counters = Counter()
    issues = Counter()
    examples = []
    for race in races:
        code = race.get('code')
        if not code:
            continue
        counters['race_cards'] += 1
        cut = race.get('cutoff')
        if race.get('basic_ready'):
            counters['csv_basic_ready_at_snapshot'] += 1
            lead = gap(cut, race.get('basic_available'))
            if lead is not None and lead >= 360:
                counters['csv_timestamp_at_least_six_minutes'] += 1
        else:
            issues.update(race.get('issues') or [])
        api = api_races.get(code, {})
        if api.get('preview_complete'):
            counters['api_complete_at_snapshot'] += 1
        if api.get('preview_complete') and not race.get('basic_ready'):
            counters['api_complete_csv_not_ready_at_snapshot'] += 1
            if len(examples) < 12:
                examples.append({'code':code, 'cutoff':cut, 'csv_issues':race.get('issues', []),
                                 'api_first_seen':api.get('first_observed_by_boat12')})
        snap = staged_races.get(code)
        if snap:
            counters['api_pre_cutoff_snapshot_recorded'] += 1
            remaining = gap(snap.get('cutoff'),snap.get('observed_at'))
            if remaining is not None and remaining >= 360:
                counters['api_observed_at_least_six_minutes_before_cutoff'] += 1
        tracked = seen_races.get(code, {})
        first = (tracked.get('first_detected') or {})
        points = []
        for key in ('sui', 'tkz', 'stt'):
            at = (first.get(key) or {}).get('at')
            if not at:
                points = []
                break
            points.append(parse(at))
        if len(points) == 3 and all(points):
            lead = gap(cut, max(points).isoformat())
            if lead is not None and lead > 0:
                counters['csv_first_seen_all_sources_before_cutoff'] += 1
                if lead >= 360:
                    counters['csv_first_seen_at_least_six_minutes'] += 1
    result = {
        'schema':'boat12-refresh-diagnostic-v1', 'created_at':datetime.now(timezone.utc).isoformat(),
        'date': day, 'conditions_generated_at':conditions.get('generated_at'),
        'openapi_checked_at':openapi.get('checked_at') if openapi.get('date')==day else None,
        'counts':dict(counters), 'csv_missing_reasons':dict(issues),
        'api_ready_csv_not_ready_examples': examples,
        'notes':[
            'Snapshot completeness counts are not live pre-cutoff availability.',
            'Provider/source availability timestamps do not prove BOAT12 first observed them.',
            'Six-minute first-seen counts depend on monitor start time and polling schedule.',
            'Staged Open API data is not inserted into prediction models.',
            'This report does not alter production conditions, predictions, or safety gates.'
        ], 'production_changed':False
    }
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'day':day, 'counts':dict(counters)},ensure_ascii=False))

if __name__ == '__main__':
    main()
