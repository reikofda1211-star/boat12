#!/usr/bin/env python3
"""Audited dataset builder for the *future* BOAT12 trifecta live model.

Only pre-deadline per-source snapshots and confirmed 1st-3rd finishing orders
are included. Does NOT alter the existing published trifecta probabilities.
"""
import argparse
import collections
import gzip
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from trifecta_live_features import extract_predeadline, _time

ROOT = Path('history/conditions/days')
LABELS = Path('history/trifecta/labels')
OUT = Path('history/trifecta/live_dataset.json')
REPORT = Path('trifecta_live_report.json')
JST = ZoneInfo('Asia/Tokyo')


def validate_record(record, labels):
    code = record.get('code')
    if not isinstance(code, str) or not re.fullmatch(r'\d{12}', code):
        return None, 'invalid_race_code'
    day = code[:8]
    cutoff = _time(record.get('cutoff'))
    available = _time(record.get('basic_available'))
    stamps = record.get('source_times') or {}
    if not cutoff or not available or not isinstance(stamps, dict):
        return None, 'missing_timestamps'
    for k in ('sui', 'tkz', 'stt'):
        snap = _time(stamps.get(k))
        if not snap or not snap <= available < cutoff or (cutoff-snap).total_seconds() > 30*60:
            return None, 'invalid_source_timing'
    if not (available <= cutoff and (cutoff-available).total_seconds() <= 30*60):
        return None, 'invalid_feature_timing'
    feature, reason = extract_predeadline(record, available.isoformat())
    if feature is None:
        return None, reason
    label = labels.get(code)
    if not isinstance(label,str) or not re.fullmatch(r'[1-6]-[1-6]-[1-6]',label):
        return None, 'missing_trifecta_result'
    rank = [int(x) for x in label.split('-')]
    if len(set(rank))!=3:
        return None,'invalid_finish'
    result_time = _time(record.get('result_recorded_at'))
    # Result must have been observed *after* the race, not before the predictors.
    if not result_time or result_time <= cutoff:
        return None,'unverified_result_timing'
    if record.get('date','').replace('-','') != day or available.astimezone(JST).strftime('%Y%m%d') != day:
        return None,'date_mismatch'
    if not 1 <= int(code[8:10]) <= 24 or not 1<=int(code[10:12])<=12:
        return None,'invalid_venue_or_race'
    return {'code':code,'day':day,'venue':int(code[8:10]),'race':int(code[10:12]),
            'cutoff':cutoff.isoformat(),'available_at':available.isoformat(),
            'finish':label,'features':feature},None


def build(root=ROOT, labels_folder=LABELS):
    labelmap={}
    for p in sorted(labels_folder.glob('????????.json')):
        try:
            obj=json.loads(p.read_text(encoding='utf-8'))
            for code,finish in obj.get('labels',{}).items():
                if code.startswith(p.stem): labelmap[code]=finish
        except (OSError, ValueError, TypeError):
            continue
    rows={}; rejects=collections.Counter()
    for p in sorted(root.glob('????????.jsonl.gz')):
        try:
            with gzip.open(p,'rt',encoding='utf-8') as handle:
                for line in handle:
                    try:
                        record=json.loads(line)
                        if not isinstance(record,dict): raise ValueError('bad record')
                        row,reason=validate_record(record,labelmap)
                        if reason: rejects[reason]+=1
                        elif row['code'] in rows: rejects['duplicate']+=1
                        else:rows[row['code']]=row
                    except (ValueError,KeyError,TypeError,OverflowError):
                        rejects['malformed_line']+=1
        except (OSError,EOFError):
            rejects['unreadable_archive']+=1
    return [rows[k] for k in sorted(rows)],dict(rejects)


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:
        from datetime import timedelta
        t=datetime(2026,10,14,10,0,tzinfo=JST)
        cutoff=t+timedelta(minutes=10)
        record={'code':'202610140101','date':'2026-10-14','cutoff':cutoff.isoformat(),'basic_available':t.isoformat(),
                'result_recorded_at':(cutoff+timedelta(minutes=20)).isoformat(),
                'source_times':{k:t.isoformat() for k in ('sui','tkz','stt')},'basic_ready':True,'original_ready':False,'cancelled':False,
                'weather':{'wind_speed':3,'wave_cm':4,'wind_direction_code':'5'},
                'players':[{'boat':i,'motor_rate':40,'boat_rate':35,'exhibition':6.8,'exhibition_st':.12,'course':i,'tilt':0} for i in range(1,7)]}
        ok,err=validate_record(record,{'202610140101':'1-2-3'})
        assert ok and not err
        record['source_times']['sui']=(cutoff+timedelta(seconds=1)).isoformat()
        assert validate_record(record,{'202610140101':'1-2-3'})[0] is None
        print('live dataset safety tests: PASS')
        return
    rows,rejected=build()
    dates=sorted({r['day'] for r in rows})
    report={'generated_at':datetime.now(JST).isoformat(),'status':'dataset_ready_for_model_research' if len(rows)>=2400 and len(dates)>=30 else 'insufficient_live_history',
            'eligible_races':len(rows),'eligible_days':len(dates),'first_day':dates[0] if dates else None,
            'last_day':dates[-1] if dates else None,'excluded':rejected,
            'message':'集計は学習・確率の公開を意味しません。過去の締切前取得日時と着順を照合しました。',
            'next_step':'必要件数を満たしたら、事前モデルとの同期間の時系列比較と確率較正を実施する。'}
    write(OUT,{'schema':'boat12-live-dataset-v1','rows':rows})
    write(REPORT,report)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
