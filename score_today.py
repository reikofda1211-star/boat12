"""BOAT12: audit previously recorded, pre-deadline 1-2 estimates vs published results.

No forecast is generated or retroactively modified. Output accuracy.json is a
transparent per-race ledger, NOT a calibration or profitability claim.
"""
from __future__ import annotations
import csv
import io
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import requests

JST = ZoneInfo('Asia/Tokyo')
VENUES = ['桐生','戸田','江戸川','平和島','多摩川','浜名湖','蒲郡','常滑','津','三国','びわこ','住之江','尼崎','鳴門','丸亀','児島','宮島','徳山','下関','若松','芦屋','福岡','唐津','大村']
RESULTS_URL = 'https://raw.githubusercontent.com/BoatraceCSV/boatracecsv.github.io/main/data/results/realtime/{year}/{month}/{day}.csv'
OUTPUT = Path('accuracy.json')


def parse_time(value):
    try:
        value = datetime.fromisoformat(value.replace('Z','+00:00'))
        return value if value.tzinfo is not None else None
    except (TypeError, ValueError, AttributeError):
        return None


def snapshots(day, folder):
    """Latest authentic pre-cutoff prediction per race; no reconstruction."""
    latest = {}
    for path in sorted(folder.glob('*.json')):
        try:
            raw = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if not isinstance(raw, list):
            continue
        for p in raw:
            if not isinstance(p, dict) or p.get('date') != day:
                continue
            code = p.get('code')
            made, cutoff = parse_time(p.get('generated_at')), parse_time(p.get('cutoff'))
            score = p.get('probability')
            if not (isinstance(code, str) and re.fullmatch(r'\d{12}',code) and code.startswith(day.replace('-',''))):
                continue
            if not (made and cutoff and made < cutoff and isinstance(score,(float,int)) and not isinstance(score,bool) and 0 <= score <= 1):
                continue
            if p.get('display_class') != 'unverified_experimental':
                continue
            current = latest.get(code)
            if current is None or made > parse_time(current['generated_at']):
                latest[code] = p
    return latest


def parse_results(text, day):
    results = {}
    for item in csv.DictReader(io.StringIO(text)):
        code = item.get('レースコード', '')
        if not re.fullmatch(r'\d{12}',code) or not code.startswith(day.replace('-','')):
            continue
        a,b=item.get('1着_艇番'),item.get('2着_艇番')
        if a not in {'1','2','3','4','5','6'} or b not in {'1','2','3','4','5','6'} or a == b:
            continue
        obtained=parse_time(item.get('取得日時'))
        if obtained is None or obtained.astimezone(JST).date().isoformat() != day:
            continue
        if code not in results or obtained > parse_time(results[code]['obtained_at']):
            results[code]={'first':int(a),'second':int(b),'obtained_at':obtained.isoformat()}
    return results


def build(day, predictions, results, checked_at, status='ok', detail=None):
    rows=[]
    for code,p in sorted(predictions.items()):
        finish=results.get(code)
        # A matching result must be recorded AFTER the saved forecast.
        if finish and parse_time(finish['obtained_at']) <= parse_time(p['generated_at']):
            finish=None
        v=int(code[8:10])
        rows.append({
            'code':code, 'venue':VENUES[v-1] if 1 <= v <= 24 else '不明', 'race':int(code[10:]),
            'generated_at':p['generated_at'], 'cutoff':p['cutoff'],
            'probability':round(p['probability'],6), 'scope':p.get('scope'),
            'result_status':'confirmed' if finish else 'pending',
            'first':finish['first'] if finish else None,
            'second':finish['second'] if finish else None,
            'hit':(finish['first']==1 and finish['second']==2) if finish else None,
            'result_recorded_at':finish['obtained_at'] if finish else None,
        })
    return {
        'schema':'boat12-accuracy-v1','date':day,'generated_at':checked_at,
        'status':status, 'message':detail,
        'source':'BoatraceCSV realtime results; forecast logs from history/conditions/predictions',
        'definition':'各レース締切前の最新保存予測のうち、閾値以上を1-2候補とし、結果確定済み候補に対する1-2決着数の割合。結果待ちは分母から除外。',
        'records':rows
    }


def main():
    now=datetime.now(JST)
    day=now.date().isoformat()
    latest=snapshots(day,Path('history/conditions/predictions') / day)
    url=RESULTS_URL.format(year=now.strftime('%Y'),month=now.strftime('%m'),day=now.strftime('%d'))
    status='ok'; message=None; results={}
    try:
        response=requests.get(url,timeout=(8,25),headers={'User-Agent':'boat12-accuracy/1.0'})
        if response.status_code == 404:
            status='results_not_published'
        elif response.status_code in (403,429):
            status='results_access_limited';message=f'HTTP {response.status_code}'
        else:
            response.raise_for_status()
            if len(response.content)>8_000_000:
                raise ValueError('Unexpectedly large results file')
            results=parse_results(response.content.decode('utf-8-sig'),day)
    except (requests.RequestException,UnicodeError,ValueError) as exc:
        status='results_fetch_failed';message=type(exc).__name__
    if status!='ok':
        # Never turn temporarily unreachable results into missing/zero outcomes.
        try:
            previous=json.loads(OUTPUT.read_text(encoding='utf-8'))
            if previous.get('date')==day and isinstance(previous.get('records'),list):
                for p in previous['records']:
                    if p.get('result_status')=='confirmed' and isinstance(p.get('code'),str):
                        results[p['code']]={'first':p['first'],'second':p['second'],'obtained_at':p['result_recorded_at']}
        except (OSError,ValueError,TypeError):
            pass
    payload=build(day,latest,results,now.isoformat(),status,message)
    temp=OUTPUT.with_suffix('.json.tmp')
    temp.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(OUTPUT)
    ready=sum(r['result_status']=='confirmed' for r in payload['records'])
    print(f'ACCURACY UPDATED {day} predictions={len(payload["records"])} confirmed={ready} status={status}',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
