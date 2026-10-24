#!/usr/bin/env python3
"""BOAT12: nationwide (24 venues) Open API monitor.

Separate from conditions.py and the prediction model. In particular, API data
without a verified acquisition timestamp is never represented as a pre-cutoff
model input. Runs once per workflow invocation and makes one API request.
"""
from __future__ import annotations
import json, os, sys, time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

JST = timezone(timedelta(hours=9))
VENUES = ['桐生','戸田','江戸川','平和島','多摩川','浜名湖','蒲郡','常滑','津','三国','びわこ','住之江','尼崎','鳴門','丸亀','児島','宮島','徳山','下関','若松','芦屋','福岡','唐津','大村']
OUT = Path('openapi-status.json')
HISTORY = Path('openapi-first-seen.json')

def now(): return datetime.now(JST)
def iso(dt): return dt.isoformat(timespec='seconds')
def obj(v): return v if isinstance(v, dict) else {}
def num(v):
    try: return float(v) if v is not None and not isinstance(v,bool) else None
    except (ValueError, TypeError): return None

def get_json(day):
    url=f'https://boatraceopenapi.github.io/api/v1/{day[:4]}/{day.replace("-", "")}.json'
    req=Request(url,headers={'User-Agent':'BOAT12-data-status/1.0','Accept':'application/json'})
    with urlopen(req, timeout=25) as res:
        body=res.read(8_000_001)
        if len(body)>8_000_000: raise ValueError('API data exceeds size limit')
        return json.loads(body),url

def parse(payload, day, previous):
    first=obj(previous.get('first_seen')) if previous.get('date') == day else {}
    all_venues={}
    data=obj(obj(payload.get('programs')).get('stadiums'))
    seen_at=iso(now())
    total=ready=0
    for venue_id in range(1,25):
        venue=VENUES[venue_id-1]
        stadium=obj(data.get(str(venue_id)))
        races=obj(stadium.get('races'))
        output=[]
        for race_id in range(1,13):
            race=obj(races.get(str(race_id)))
            if not race: continue
            if race.get('date') != day or int(race.get('stadium_number',-1)) != venue_id or int(race.get('race_number',-1)) != race_id:
                continue
            preview=obj(race.get('preview'))
            racers=obj(preview.get('racers'))
            basic=all(num(preview.get(k)) is not None for k in ('wind_speed','wave_height'))
            six=all(isinstance(racers.get(str(i)),dict) and num(racers[str(i)].get('exhibition_time')) is not None and num(racers[str(i)].get('start_timing')) is not None and num(racers[str(i)].get('course_number')) is not None for i in range(1,7))
            complete=bool(basic and six and len(racers)==6)
            key=f'{day.replace("-", "")}{venue_id:02d}{race_id:02d}'
            if complete and key not in first: first[key]=seen_at
            output.append({'code':key,'race':race_id,'cutoff':race.get('closed_at'), 'preview_present':bool(preview),
                           'weather_present':basic,'six_boats_present':six,'preview_complete':complete,
                           'first_observed_by_boat12':first.get(key),
                           'wind_speed':preview.get('wind_speed') if complete else None,
                           'wave_height':preview.get('wave_height') if complete else None})
            total+=1;ready+=int(complete)
        all_venues[f'{venue_id:02d}']={'name':venue,'races':output,'count':len(output),'ready':sum(r['preview_complete'] for r in output)}
    result={'schema':'boat12-openapi-monitor-v1','date':day,'checked_at':seen_at,'source':'Boatrace Open API (unofficial)',
            'venue_count':24,'race_count':total,'complete_previews':ready,'venues':all_venues,
            'disclaimer':'First observed is BOAT12 GitHub Actions detection time, NOT original collection/publication time. Do not use this alone as proof of pre-deadline availability.'}
    return result,{'date':day,'first_seen':first}

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    os.replace(temp,path)

def main():
    day=now().date().isoformat()
    previous=json.loads(HISTORY.read_text(encoding='utf-8')) if HISTORY.exists() else {}
    try:
        payload,url=get_json(day)
        result,history=parse(payload,day,previous)
        result['source_url']=url
        write(OUT,result);write(HISTORY,history)
        print('OPEN API STATUS',day,'races',result['race_count'],'complete',result['complete_previews'])
    except (HTTPError,URLError,ValueError,KeyError,TypeError,TimeoutError,json.JSONDecodeError) as e:
        write(OUT,{'schema':'boat12-openapi-monitor-v1','date':day,'checked_at':iso(now()),'status':'unavailable', 'reason':type(e).__name__,
                   'message':'API data not available; existing BOAT12 prediction feed not changed.'})
        print('OPEN API UNAVAILABLE',type(e).__name__,file=sys.stderr)
        return 0
    return 0
if __name__=='__main__':sys.exit(main())
