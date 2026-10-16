"""Record first detection of pre-deadline data from existing conditions.json.

Does not fetch, predict or change model release criteria. Provider timestamps
are NOT guaranteed publication times; this logs BOAT12's first *observed* time.
"""
from __future__ import annotations
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

JST = ZoneInfo('Asia/Tokyo')
DATA = Path('conditions.json')
ROOT = Path('history/conditions/latency')
KINDS = ('tkz', 'stt', 'sui', 'original')

def parsed(s):
    try:
        d=datetime.fromisoformat(str(s).replace('Z','+00:00'))
        return d.astimezone(JST) if d.tzinfo else None
    except (ValueError, TypeError):
        return None

def run():
    if not DATA.exists():
        print('LATENCY AUDIT: conditions.json not found; no changes')
        return
    doc=json.loads(DATA.read_text(encoding='utf-8'))
    date=doc.get('date')
    now=datetime.now(JST)
    generated=parsed(doc.get('generated_at'))
    if not generated or date != now.date().isoformat() or not isinstance(doc.get('races'),list):
        print('LATENCY AUDIT: current-day data not available; skipped')
        return
    ROOT.mkdir(parents=True,exist_ok=True)
    out=ROOT / f'{date}.json'
    old={}
    if out.exists():
        try:old=json.loads(out.read_text(encoding='utf-8'))
        except (ValueError,OSError):old={}
    seen=old.get('races',{}) if isinstance(old.get('races'),dict) else {}
    counts={'pending':0,'basic_ready':0,'original_ready':0,'experimental':0}
    for race in doc['races']:
        code=race.get('code'); cutoff=parsed(race.get('cutoff'))
        if not isinstance(code,str) or len(code)!=12 or not code.startswith(date.replace('-','')):
            continue
        entry=seen.setdefault(code,{'venue':race.get('venue'),'race':race.get('race'),'cutoff':race.get('cutoff'),'first_detected':{}})
        first=entry.setdefault('first_detected',{})
        for kind in KINDS:
            source=(race.get('source_times') or {}).get(kind)
            when=parsed(source)
            # This records first detection by BOAT12, not provider's publication time.
            if when and when<=generated and kind not in first:
                first[kind]={'at':generated.isoformat(),'provider_recorded_at':source,'before_cutoff':bool(cutoff and generated<cutoff)}
        obs=parsed((race.get('weather') or {}).get('observed_at'))
        if obs and 'weather_observation' not in first:
            first['weather_observation']={'at':generated.isoformat(),'provider_recorded_at':obs.isoformat(),'before_cutoff':bool(cutoff and generated<cutoff)}
        # Latest state retained as a diagnostic; historical first_seen is never overwritten.
        entry['last_seen_at']=generated.isoformat()
        entry['last_state']=race.get('state')
        entry['last_issues']=race.get('issues',[])
        entry['basic_ready']=bool(race.get('basic_ready'))
        entry['original_ready']=bool(race.get('original_ready'))
        entry['experimental_recorded']=bool(race.get('experimental_prediction'))
        if cutoff and cutoff>generated:
            counts['pending']+=1
        counts['basic_ready']+=bool(race.get('basic_ready'))
        counts['original_ready']+=bool(race.get('original_ready'))
        counts['experimental']+=bool(race.get('experimental_prediction'))
    result={'schema':'boat12-latency-audit-v1','date':date,'checked_at':now.isoformat(),
            'conditions_generated_at':generated.isoformat(),'counts':counts,'races':seen,
            'notice':'first_detected.at is first appearance in BOAT12 generated data, not public release time; checks occur only during Actions runs.'}
    temp=out.with_suffix('.json.tmp')
    temp.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(temp,out)
    print('LATENCY AUDIT:',date,'races=',len(seen),'first observations recorded')

if __name__=='__main__':
    run()
