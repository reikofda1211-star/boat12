#!/usr/bin/env python3
"""Read-only BOAT12 OpenAPI/conditions compatibility audit.

Does not modify conditions.json, release gates, models, or published forecasts.
OpenAPI observations are historical first-seen snapshots; no source publication
or exact weather-observation time is inferred.
"""
from __future__ import annotations
import argparse
import json
import math
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

JST = __import__('zoneinfo').ZoneInfo('Asia/Tokyo')

def time_of(s):
    if not isinstance(s, str): return None
    try:
        v = datetime.fromisoformat(s.replace('Z', '+00:00'))
        return v.astimezone(JST) if v.tzinfo else None
    except ValueError: return None

def finite(value, lo, hi):
    if isinstance(value, bool): return None
    try: n = float(value)
    except (TypeError, ValueError): return None
    return n if math.isfinite(n) and lo <= n <= hi else None

def compare(staged, conditions):
    output = {'schema':'boat12-openapi-adapter-audit-v1', 'date':staged.get('date'),
              'mode':'read_only_no_predictions', 'counts':{}, 'records':{},
              'note':'First-seen time is BOAT12 observation, NOT source publication time; precise meteorological observation time is unavailable.'}
    counts = Counter()
    if staged.get('schema') != 'boat12-openapi-precutoff-v1' or staged.get('date') != conditions.get('date'):
        output['error'] = 'schema_or_date_mismatch'; return output
    refmap = {r.get('code'):r for r in conditions.get('races',[]) if isinstance(r,dict)}
    for code, snapshot in (staged.get('races') or {}).items():
        counts['staged'] += 1
        reasons = []
        ref = refmap.get(code)
        seen, cutoff = time_of(snapshot.get('observed_at')), time_of(snapshot.get('cutoff'))
        if not ref: reasons.append('missing_trusted_race_card')
        elif time_of(ref.get('cutoff')) != cutoff: reasons.append('cutoff_mismatch')
        if not seen or not cutoff or seen >= cutoff - timedelta(seconds=30): reasons.append('not_safely_before_cutoff')
        if not snapshot.get('registration_ids_verified'): reasons.append('registration_ids_not_verified')
        sp = snapshot.get('players')
        if not isinstance(sp, list) or len(sp) != 6: reasons.append('not_six_boats')
        else:
            courses = []
            for i,p in enumerate(sp,1):
                if not isinstance(p,dict) or p.get('boat') != i: reasons.append('boat_order_mismatch');break
                if finite(p.get('exhibition'),4,12) is None: reasons.append('missing_exhibition')
                if finite(p.get('exhibition_st'),-1,2) is None: reasons.append('missing_exhibition_st')
                course = finite(p.get('course'),1,6)
                if course is None or int(course) != course: reasons.append('missing_course')
                else: courses.append(int(course))
            if sorted(courses) != list(range(1,7)): reasons.append('invalid_course_permutation')
        if finite(snapshot.get('wind_speed'),0,40) is None: reasons.append('missing_wind_speed')
        if finite(snapshot.get('wave_cm'),0,200) is None: reasons.append('missing_wave_height')
        # These are critical model compatibility limitations.
        if snapshot.get('wind_direction_code') is None: reasons.append('missing_wind_direction')
        if not snapshot.get('weather_code'): reasons.append('missing_weather_code')
        # Never impute the true observation time using collection time.
        reasons.append('weather_observation_time_unverified')
        reasons.append('original_exhibition_not_in_openapi_snapshot')
        if not reasons: counts['direct_ready'] += 1
        else: counts['needs_adapter_or_validation'] += 1
        if ref and not any(x in reasons for x in ('missing_trusted_race_card','cutoff_mismatch','not_safely_before_cutoff','registration_ids_not_verified','not_six_boats','boat_order_mismatch','missing_exhibition','missing_exhibition_st','missing_course','invalid_course_permutation','missing_wind_speed','missing_wave_height')):
            counts['basic_fields_crosschecked'] += 1
        output['records'][code]={'venue':ref.get('venue') if ref else None,
                                 'race':ref.get('race') if ref else None,
                                 'first_observed':snapshot.get('observed_at'),
                                 'cutoff':snapshot.get('cutoff'),
                                 'source':'Boatrace Open API (unofficial)',
                                 'compatible_basic_fields':ref is not None and not any(x in reasons for x in ('missing_trusted_race_card','cutoff_mismatch','not_safely_before_cutoff','registration_ids_not_verified','not_six_boats','boat_order_mismatch','missing_exhibition','missing_exhibition_st','missing_course','invalid_course_permutation','missing_wind_speed','missing_wave_height')),
                                 'eligible_for_existing_model':False,
                                 'reasons':sorted(set(reasons))}
    output['counts'] = dict(counts)
    return output

def selftest():
    day='2026-10-24'; code='202610240508'; cutoff='2026-10-24T13:53:00+09:00'
    players=[{'boat':i,'exhibition':6.5,'exhibition_st':0.1,'course':i} for i in range(1,7)]
    s={'schema':'boat12-openapi-precutoff-v1','date':day,'races':{code:{'observed_at':'2026-10-24T13:44:45+09:00','cutoff':cutoff,'registration_ids_verified':True,'players':players,'wind_speed':1,'wave_cm':2,'wind_direction_code':5,'weather_code':3}}}
    c={'date':day,'races':[{'code':code,'cutoff':cutoff,'venue':'多摩川','race':8}]}
    d=compare(s,c)
    assert d['counts']['basic_fields_crosschecked']==1
    assert d['records'][code]['eligible_for_existing_model'] is False
    s['races'][code]['observed_at']='2026-10-24T13:54:45+09:00'
    assert 'not_safely_before_cutoff' in compare(s,c)['records'][code]['reasons']
    print('SELF TEST PASSED')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--staged',default='openapi-precutoff.json')
    ap.add_argument('--conditions',default='conditions.json')
    ap.add_argument('--out',default='openapi-adapter-audit.json')
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test: selftest(); return
    staged=json.loads(Path(args.staged).read_text(encoding='utf-8'))
    conditions=json.loads(Path(args.conditions).read_text(encoding='utf-8'))
    result=compare(staged,conditions)
    Path(args.out).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'counts':result['counts'],'error':result.get('error')},ensure_ascii=False))

if __name__=='__main__': main()
