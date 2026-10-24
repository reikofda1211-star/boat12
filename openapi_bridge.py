#!/usr/bin/env python3
"""BOAT12 safe OpenAPI snapshot staging; NEVER feeds models automatically.

Run alongside existing monitor on its 5-minute schedule. Each run fetches one
nationwide JSON and stores only races witnessed before the known cutoff, whose
six registration IDs agree with BOAT12's trusted race cards. No retroactive
pre-deadline claims are made and post-cutoff previews never enter the staging.
"""
import json, os, urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

JST=timezone(timedelta(hours=9))
OUT=Path('openapi-precutoff.json')
CONDITIONS=Path('conditions.json')


def get_time(value):
    try:
        text=str(value).strip()
        if len(text)==19 and text[10]==' ':
            return datetime.strptime(text,'%Y-%m-%d %H:%M:%S').replace(tzinfo=JST)
        x=datetime.fromisoformat(text.replace('Z','+00:00'))
        return x.astimezone(JST) if x.tzinfo else None
    except (TypeError,ValueError):
        return None


def finite_number(x, minimum, maximum):
    try:
        n=float(x)
        return n if n == n and minimum<=n<=maximum else None
    except (TypeError,ValueError):
        return None


def atomic_json(path, obj):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temp,path)


def run():
    now=datetime.now(JST)
    day=now.date().isoformat()
    old=json.loads(OUT.read_text(encoding='utf-8')) if OUT.exists() else {}
    kept=old.get('races',{}) if old.get('date')==day and isinstance(old.get('races'),dict) else {}
    # Cards joined by registration, never by racer name alone.
    conditions=json.loads(CONDITIONS.read_text(encoding='utf-8')) if CONDITIONS.exists() else {}
    cards={r.get('code'):r for r in conditions.get('races',[]) if conditions.get('date')==day}
    url=f'https://boatraceopenapi.github.io/api/v1/{day[:4]}/{day.replace("-", "")}.json'
    request=urllib.request.Request(url,headers={'User-Agent':'BOAT12-precutoff-research/1.0','Accept':'application/json'})
    with urllib.request.urlopen(request,timeout=25) as response:
        content=response.read(8_000_001)
        if len(content)>8_000_000: raise ValueError('API response too large')
    data=json.loads(content)
    stadiums=data.get('programs',{}).get('stadiums',{})
    skipped={'no_card':0,'wrong_ids':0,'incomplete':0,'deadline':0}
    for venue in range(1,25):
        races=stadiums.get(str(venue),{}).get('races',{})
        for race_no in range(1,13):
            entry=races.get(str(race_no),{})
            if not isinstance(entry,dict) or entry.get('date') != day: continue
            code=f'{day.replace("-","")}{venue:02d}{race_no:02d}'
            if code in kept: continue  # Immutable first observed snapshot.
            ref=cards.get(code)
            if not ref or len(ref.get('players',[]))!=6:
                skipped['no_card']+=1;continue
            cutoff=get_time(entry.get('closed_at'))
            ref_cutoff=get_time(ref.get('cutoff'))
            if cutoff is None or ref_cutoff != cutoff or now>=cutoff-timedelta(seconds=30):
                skipped['deadline']+=1;continue
            racers=entry.get('racers') or {}
            preview=entry.get('preview') or {}
            entries=preview.get('racers') or {}
            if not isinstance(racers,dict) or not isinstance(entries,dict):
                skipped['incomplete']+=1;continue
            try:
                matches=all(int(racers[str(i)]['number'])==int(ref['players'][i-1]['registration']) for i in range(1,7))
            except (TypeError,ValueError,KeyError,IndexError): matches=False
            if not matches:
                skipped['wrong_ids']+=1;continue
            ws=finite_number(preview.get('wind_speed'),0,40)
            wave=finite_number(preview.get('wave_height'),0,200)
            boats=[]
            for i in range(1,7):
                p=entries.get(str(i),{})
                if not isinstance(p,dict):break
                exhibition=finite_number(p.get('exhibition_time'),4,12)
                start=finite_number(p.get('start_timing'),-1,2)
                course=finite_number(p.get('course_number'),1,6)
                tilt=finite_number(p.get('tilt_adjustment'),-1,4)
                if exhibition is None or start is None or course is None or int(course)!=course:break
                boats.append({'boat':i,'exhibition':exhibition,'exhibition_st':start,'course':int(course),'tilt':tilt})
            if ws is None or wave is None or len(boats)!=6 or sorted(b['course'] for b in boats)!=list(range(1,7)):
                skipped['incomplete']+=1;continue
            kept[code]={'code':code,'observed_at':now.isoformat(timespec='seconds'),
                        'cutoff':cutoff.isoformat(),'source':'Boatrace Open API (unofficial)',
                        'registration_ids_verified':True,'wind_speed':ws,'wave_cm':wave,
                        'wind_direction_code':preview.get('wind_direction_number'),
                        'weather_code':preview.get('weather_number'),
                        'air_c':finite_number(preview.get('air_temperature'),-20,55),
                        'water_c':finite_number(preview.get('water_temperature'),-5,45),
                        'players':boats,
                        'warning':'Staged research data only. Not model-validated or release-eligible.'}
    atomic_json(OUT,{'schema':'boat12-openapi-precutoff-v1','date':day,
                     'checked_at':now.isoformat(timespec='seconds'),
                     'races':kept,'count':len(kept),'skipped_last_run':skipped,
                     'notice':'First witnessed before cutoff on this GitHub runner. No claim of origin publication time. Never auto-injected into conditions.json.'})
    print('OPENAPI PRE-CUTOFF STAGED:',len(kept),'races; skipped:',skipped)

if __name__=='__main__':
    run()
