#!/usr/bin/env python3
"""BOAT12 independent weather comparison: non-predictive, read-only inputs.

Compare OpenAPI's first observed pre-cutoff values against BOAT12/BoatraceCSV
weather snapshots. Matching numbers DO NOT establish identical observations
or prove OpenAPI's weather observation timestamp. Does not modify forecasts.
"""
from __future__ import annotations
import argparse
import json
import math
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path


def dt(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return parsed if parsed.tzinfo else None
    except ValueError:
        return None


def number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def compare(staged, conditions):
    result = {
        'schema': 'boat12-weather-crosscheck-v1',
        'date': staged.get('date'),
        'mode': 'read_only_comparison_no_model_use',
        'counts': {}, 'races': {},
        'notice': ('Wind/wave value matches are not proof of matching weather observation times. '
                   'OpenAPI has no independently verified meteorological observation timestamp. '
                   'No predictions or release rules are changed.'),
    }
    if staged.get('schema') != 'boat12-openapi-precutoff-v1' or staged.get('date') != conditions.get('date'):
        result['error'] = 'schema_or_date_mismatch'
        return result
    refs = {x.get('code'): x for x in conditions.get('races', []) if isinstance(x, dict)}
    counts = Counter()
    for code, value in (staged.get('races') or {}).items():
        if not isinstance(value, dict):
            continue
        counts['staged_races'] += 1
        ref = refs.get(code)
        reasons = []
        received = dt(value.get('observed_at'))
        cutoff = dt(value.get('cutoff'))
        if ref is None:
            reasons.append('boat12_record_missing')
        elif dt(ref.get('cutoff')) != cutoff:
            reasons.append('cutoff_mismatch')
        if not received or not cutoff or received >= cutoff - timedelta(seconds=30):
            reasons.append('openapi_not_verified_pre_cutoff')
        if value.get('registration_ids_verified') is not True:
            reasons.append('registration_ids_not_verified')

        other = (ref or {}).get('weather') or {}
        pairs = {'wind_speed_mps': (value.get('wind_speed'), other.get('wind_speed')),
                 'wave_height_cm': (value.get('wave_cm'), other.get('wave_cm')),
                 'air_c': (value.get('air_c'), other.get('air_c')),
                 'water_c': (value.get('water_c'), other.get('water_c'))}
        compared = {}
        for name, (open_value, csv_value) in pairs.items():
            o, c = number(open_value), number(csv_value)
            status = 'missing' if o is None or c is None else 'match' if abs(o-c) < 1e-9 else 'mismatch'
            compared[name] = {'openapi': o, 'boatracecsv': c, 'comparison': status}
        vital = [compared['wind_speed_mps']['comparison'], compared['wave_height_cm']['comparison']]
        if 'missing' in vital:
            state = 'missing_weather_values'
        elif 'mismatch' in vital:
            state = 'wind_or_wave_mismatch'
        else:
            state = 'wind_and_wave_match'
        counts[state] += 1

        csv_observation = dt(other.get('observed_at'))
        csv_source = dt(((ref or {}).get('source_times') or {}).get('sui'))
        if csv_observation is None:
            reasons.append('boatracecsv_observation_time_missing')
        if csv_source is None:
            reasons.append('boatracecsv_source_time_missing')
        if csv_source and cutoff and csv_source >= cutoff:
            reasons.append('boatracecsv_source_not_pre_cutoff')
        if csv_observation and csv_source and csv_observation > csv_source:
            reasons.append('csv_observation_after_source_capture')
        # The OpenAPI weather instrument observation time is unavailable even when numbers match.
        reasons.append('openapi_weather_observation_time_unverified')
        result['races'][code] = {
            'venue': (ref or {}).get('venue'), 'race': (ref or {}).get('race'),
            'cutoff': value.get('cutoff'), 'openapi_received_at': value.get('observed_at'),
            'boatracecsv_weather_observed_at': other.get('observed_at'),
            'boatracecsv_source_captured_at': ((ref or {}).get('source_times') or {}).get('sui'),
            'wind_wave_status': state, 'fields': compared,
            'weather_timestamp_equivalence_proven': False,
            'safe_for_automatic_model_injection': False,
            'reasons': sorted(set(reasons)),
        }
    result['counts'] = dict(counts)
    return result


def selftest():
    code = '202610240508'
    cutoff = '2026-10-24T13:53:00+09:00'
    stage = {'schema':'boat12-openapi-precutoff-v1','date':'2026-10-24',
             'races':{code:{'cutoff':cutoff,'observed_at':'2026-10-24T13:44:45+09:00',
                            'registration_ids_verified':True,'wind_speed':1,'wave_cm':2}}}
    base = {'date':'2026-10-24','races':[{'code':code,'cutoff':cutoff,'venue':'多摩川','race':8,
             'weather':{'wind_speed':1,'wave_cm':2,'observed_at':'2026-10-24T13:27:00+09:00'},
             'source_times':{'sui':'2026-10-24T13:45:30+09:00'}}]}
    x = compare(stage, base)
    assert x['counts']['wind_and_wave_match'] == 1
    assert not x['races'][code]['safe_for_automatic_model_injection']
    stage['races'][code]['wave_cm'] = 3
    assert compare(stage, base)['counts']['wind_or_wave_mismatch'] == 1
    stage['races'][code]['wave_cm'] = None
    assert compare(stage, base)['counts']['missing_weather_values'] == 1
    print('SELF TEST PASSED')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--staged', default='openapi-precutoff.json')
    parser.add_argument('--conditions', default='conditions.json')
    parser.add_argument('--out', default='openapi-weather-audit.json')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        selftest()
        return
    staged = json.loads(Path(args.staged).read_text(encoding='utf-8'))
    conditions = json.loads(Path(args.conditions).read_text(encoding='utf-8'))
    output = compare(staged, conditions)
    Path(args.out).write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'counts':output['counts'],'error':output.get('error')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
