#!/usr/bin/env python3
"""Separate recent-race training for boat12 3連単 research; does not modify 1-2 files."""
import csv, io, json, re, sys, time
from datetime import timedelta
from pathlib import Path
import requests
import trifecta_train as base

CACHE=Path('history/trifecta/recent_training.json')
METRIC_MAP={'win':'全国勝率','rate':'全国2連対率','local_win':'当地勝率',
'local_rate':'当地2連対率','motor_rate':'モーター2連対率','boat_rate':'ボート2連対率'}
ROOT='https://raw.githubusercontent.com/BoatraceCSV/boatracecsv.github.io/main/data'

def getcsv(s,url):
    r=s.get(url,timeout=(8,22))
    if r.status_code in (401,403,429): raise PermissionError('データ取得元がアクセスを制限: '+str(r.status_code))
    if r.status_code==404: return None
    r.raise_for_status()
    if len(r.content)>9_000_000: raise ValueError('Oversized CSV')
    return list(csv.DictReader(io.StringIO(r.content.decode('utf-8-sig'))))

def fetch_day(s,day):
    yyyy,mm,dd=day[:4],day[4:6],day[6:8]
    suffix=f'{yyyy}/{mm}/{dd}.csv'
    cards=getcsv(s,ROOT+'/programs/race_cards/'+suffix)
    results=getcsv(s,ROOT+'/results/realtime/'+suffix)
    if not cards or not results: return None
    outcome={}
    for r in results:
        code=r.get('レースコード','')
        first=[r.get(f'{k}着_艇番','') for k in (1,2,3)]
        if not re.fullmatch(day+r'\d{4}',code) or any(x not in '123456' or len(x)!=1 for x in first) or len(set(first))!=3:continue
        # Only take recorded post-race results; don't use fields from results as predictors.
        outcome[code]='-'.join(first)
    records=[];labels={}
    for card in cards:
        code=card.get('レースコード','')
        if code not in outcome:continue
        try:
            venue=base.VENUES[int(code[8:10])-1];race=str(int(code[10:12]))
            row={'date':day,'venue':venue,'race':race,'hit12':str(int(outcome[code].startswith('1-2-')))}
            for i in range(1,7):
                if not card.get(f'艇{i}_登録番号'):raise ValueError('missing competitor')
                for metric,col in METRIC_MAP.items(): row[f'b{i}_{metric}']=card[f'艇{i}_{col}']
            base.numbers(row)  # strict range/finite checks
            records.append(row);labels[code]=outcome[code]
        except (KeyError,ValueError,TypeError,IndexError):continue
    return records,labels

def run(max_days=30):
    old=base.read_json(CACHE)
    existing=old.get('rows',{}) if isinstance(old,dict) else {}
    if not isinstance(existing,dict): existing={}
    today=base.now().date()
    # Visit recent dates first; previous cached days are skipped, capped requests per run.
    attempted=0
    with requests.Session() as session:
        session.headers.update({'User-Agent':'boat12-trifecta-research/1.1'})
        for age in range(1,120):
            if attempted>=max_days:break
            day=(today-timedelta(days=age)).strftime('%Y%m%d')
            if day in existing:continue
            attempted+=1
            try:
                item=fetch_day(session,day)
                if item is None:
                    print(day,'source unavailable',flush=True);continue
                records,labels=item
                if len(records)<3:print(day,'insufficient valid races',flush=True);continue
                existing[day]=records
                base.write_json(base.CACHE/(day+'.json'),{'schema':base.VERSION,'day':day,'source':ROOT,
                                'obtained_at':base.now().isoformat(),'labels':labels})
                base.write_json(CACHE,{'version':'recent-v1','rows':existing})
                print(day,'saved',len(records),'races',flush=True)
            except PermissionError:raise
            except (requests.RequestException,UnicodeError,ValueError,csv.Error) as exc:
                print(day,'unavailable:',type(exc).__name__,str(exc)[:110],flush=True)
            time.sleep(.5)
    training,excluded=base.load_training()
    for day,dayrows in existing.items():
        for r in dayrows:
            try:
                key=base.key_of(r)
                if key[0]<today.strftime('%Y%m%d') and r.get('hit12') in ('0','1'):
                    base.numbers(r)
                    training[key]=r
            except (ValueError,TypeError,KeyError):continue
    rows,extra=base.joined_rows(training)
    excluded.update(extra)
    model=base.fit_model(rows,excluded)
    base.write_json(base.OUTPUT,model)
    print('RESULT',model.get('status'),model.get('message'),flush=True)
    print('RECENT_DAYS',len(existing),'VALID_ROWS',len(rows),flush=True)
    if model.get('report'):
        print('EVIDENCE_END',model.get('evidence_end'),'TOP5_TEST',model['report']['model']['top5_hit_rate'],flush=True)

if __name__=='__main__':
    n=int(sys.argv[1]) if len(sys.argv)>1 else 30
    if not 0<=n<=60:raise SystemExit('history_days 0..60')
    run(n)
