"""BOAT12: auditable actual 1-2 outcomes for saved pre-deadline forecasts.

Writes today's legacy accuracy.json (v1) and accuracy_history.json (v2).
Stores verified result snapshots separately for reuse. Does not invent backtests,
change forecast probabilities, or count missing results as losses.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

JST = ZoneInfo('Asia/Tokyo')
VENUES = ['桐生','戸田','江戸川','平和島','多摩川','浜名湖','蒲郡','常滑','津','三国','びわこ','住之江','尼崎','鳴門','丸亀','児島','宮島','徳山','下関','若松','芦屋','福岡','唐津','大村']
ROOT = Path('history/conditions/predictions')
CACHE = Path('history/accuracy_results')
TODAY = Path('accuracy.json')
HISTORY = Path('accuracy_history.json')
RESULT_URL = 'https://raw.githubusercontent.com/BoatraceCSV/boatracecsv.github.io/main/data/results/realtime/{year}/{month}/{day}.csv'
MAX_DAYS = 3650
MAX_OLD_REQUESTS = 3


def parse_time(value):
    if not isinstance(value, str):
        return None
    try:
        t = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return t if t.tzinfo else None
    except ValueError:
        return None


def load_json(path, default):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def snapshots(day, folder):
    """Take latest authentic and timely model output PER race, not later updates."""
    latest = {}
    for path in sorted(folder.glob('*.json')):
        raw = load_json(path, [])
        if not isinstance(raw, list):
            continue
        for p in raw:
            if not isinstance(p, dict) or p.get('date') != day:
                continue
            code, prob = p.get('code'), p.get('probability')
            made, cutoff = parse_time(p.get('generated_at')), parse_time(p.get('cutoff'))
            if not (isinstance(code, str) and re.fullmatch(r'\d{12}', code) and code.startswith(day.replace('-', ''))):
                continue
            if not (made and cutoff and made < cutoff and made.astimezone(JST).date().isoformat() == day):
                continue
            if not (type(prob) in (int, float) and math.isfinite(prob) and 0 <= prob <= 1):
                continue
            if p.get('display_class') != 'unverified_experimental':
                continue
            old = latest.get(code)
            if old is None or made > parse_time(old['generated_at']):
                latest[code] = p
    return latest


def parse_results(text, day):
    output = {}
    for item in csv.DictReader(io.StringIO(text)):
        code = item.get('レースコード', '')
        if not re.fullmatch(r'\d{12}', code) or not code.startswith(day.replace('-', '')):
            continue
        a, b = item.get('1着_艇番'), item.get('2着_艇番')
        if a not in set('123456') or b not in set('123456') or a == b:
            continue
        when = parse_time(item.get('取得日時'))
        if not when or when.astimezone(JST).date().isoformat() != day:
            continue
        old = output.get(code)
        if old is None or when > parse_time(old['obtained_at']):
            output[code] = {'first': int(a), 'second': int(b), 'obtained_at': when.isoformat()}
    return output


def valid_results(raw, day):
    """Only trusted result shape from provider or our own on-disk archive."""
    found = {}
    if not isinstance(raw, dict):
        return found
    for code, r in raw.items():
        if not isinstance(code, str) or not re.fullmatch(r'\d{12}', code) or not code.startswith(day.replace('-', '')) or not isinstance(r, dict):
            continue
        when = parse_time(r.get('obtained_at'))
        if type(r.get('first')) is not int or type(r.get('second')) is not int or r['first'] == r['second'] or not all(1 <= r[k] <= 6 for k in ('first', 'second')):
            continue
        if when and when.astimezone(JST).date().isoformat() == day:
            found[code] = {'first': r['first'], 'second': r['second'], 'obtained_at': when.isoformat()}
    return found


def fetch_results(day, session):
    y, m, d = day.split('-')
    url = RESULT_URL.format(year=y, month=m, day=d)
    try:
        response = session.get(url, timeout=(8, 25), headers={'User-Agent':'boat12-accuracy/2.0'})
        if response.status_code == 404:
            return {}, 'results_not_published'
        if response.status_code in (403, 429):
            return {}, 'results_access_limited'
        response.raise_for_status()
        if len(response.content) > 8_000_000:
            raise ValueError('Unexpectedly large results file')
        return parse_results(response.content.decode('utf-8-sig'), day), 'ok'
    except (requests.RequestException, UnicodeError, ValueError, csv.Error):
        return {}, 'results_fetch_failed'


def build(day, predictions, results, checked_at, status='ok'):
    rows = []
    for code, p in sorted(predictions.items()):
        result = results.get(code)
        predicted = parse_time(p['generated_at'])
        cutoff = parse_time(p['cutoff'])
        # A result must be a post-cutoff observation, not a pre-race artefact.
        if result and (parse_time(result['obtained_at']) <= predicted or parse_time(result['obtained_at']) <= cutoff):
            result = None
        venue = int(code[8:10])
        rows.append({
            'code': code, 'date': day, 'venue': VENUES[venue - 1] if 1 <= venue <= 24 else '不明',
            'race': int(code[10:12]), 'generated_at': p['generated_at'], 'cutoff': p['cutoff'],
            'probability': round(float(p['probability']), 6), 'scope': p.get('scope'),
            'result_status': 'confirmed' if result else 'pending',
            'first': result['first'] if result else None, 'second': result['second'] if result else None,
            'hit': (result['first'] == 1 and result['second'] == 2) if result else None,
            'result_recorded_at': result['obtained_at'] if result else None,
        })
    return rows


def listed_dates(today):
    days = []
    if not ROOT.exists():
        return days
    for folder in ROOT.iterdir():
        if not folder.is_dir():
            continue
        try:
            d = date.fromisoformat(folder.name)
        except ValueError:
            continue
        if today - timedelta(days=MAX_DAYS - 1) <= d <= today and any(folder.glob('*.json')):
            days.append(folder.name)
    return sorted(days, reverse=True)


def choose_fetches(days, predictions_by_day, archive_by_day, today, previous_history):
    """Fetch today, then at most 3 old days needing confirmation, avoiding overfetch."""
    chosen = [today]
    backlog = []
    for day in days:
        if day == today:
            continue
        rows = previous_history.get(day, [])
        completed = {r['code'] for r in rows if r.get('result_status') == 'confirmed'}
        predictions = predictions_by_day[day]
        if not predictions or all(code in completed or code in archive_by_day[day] for code in predictions):
            continue
        backlog.append(day)
    chosen.extend(backlog[:MAX_OLD_REQUESTS])
    return set(chosen)


def score_day(rows, threshold):
    candidates = [r for r in rows if r['probability'] >= threshold]
    settled = [r for r in candidates if r['result_status'] == 'confirmed']
    hits = sum(r['hit'] is True for r in settled)
    return {'candidates': len(candidates), 'confirmed': len(settled), 'pending': len(candidates)-len(settled),
            'hits': hits, 'rate': round(hits / len(settled), 6) if settled else None}


def main():
    now = datetime.now(JST)
    today = now.date().isoformat()
    days = listed_dates(now.date())
    if today not in days:
        days.insert(0, today)
    predictions = {d: snapshots(d, ROOT / d) for d in days}
    archive = {d: valid_results(load_json(CACHE / (d + '.json'), {}), d) for d in days}
    old_today = load_json(TODAY, {})
    old_history = load_json(HISTORY, {})
    old_rows_by_day = {}
    for r in old_history.get('records', []) if isinstance(old_history, dict) else []:
        if isinstance(r, dict) and r.get('date') in predictions and isinstance(r.get('code'), str):
            old_rows_by_day.setdefault(r['date'], []).append(r)
    if old_today.get('date') == today:
        old_rows_by_day.setdefault(today, []).extend(old_today.get('records', []))
    # Preserve already confirmed outcomes through temporary source failures.
    for d, rows in old_rows_by_day.items():
        for r in rows:
            if r.get('result_status') == 'confirmed' and r.get('code') in predictions[d]:
                candidate = valid_results({r['code']:{'first':r.get('first'),'second':r.get('second'),'obtained_at':r.get('result_recorded_at')}}, d)
                archive[d].update({code: value for code, value in candidate.items() if code not in archive[d]})
    targets = choose_fetches(days, predictions, archive, today, old_rows_by_day)
    statuses = {}
    with requests.Session() as session:
        for d in days:
            if d not in targets:
                statuses[d] = 'cached'
                continue
            retrieved, status = fetch_results(d, session)
            statuses[d] = status
            for code, r in retrieved.items():
                old = archive[d].get(code)
                if old is None or parse_time(r['obtained_at']) > parse_time(old['obtained_at']):
                    archive[d][code] = r
            if retrieved or archive[d]:
                write_json(CACHE / (d + '.json'), archive[d])
            if status == 'results_access_limited':
                # Stop accessing the same upstream source within this run.
                break
    all_rows = []
    for d in days:
        all_rows.extend(build(d, predictions[d], archive[d], now.isoformat(), statuses.get(d, 'not_checked')))
    all_rows.sort(key=lambda r:(r['date'], r['code']), reverse=True)
    today_rows = [r for r in all_rows if r['date'] == today]
    write_json(TODAY, {'schema':'boat12-accuracy-v1','date':today,'generated_at':now.isoformat(),
        'status':statuses.get(today,'not_checked'), 'message':None,
        'source':'BoatraceCSV realtime results; immutable pre-deadline forecast logs',
        'definition':'閾値以上の締切前保存予測のうち、結果確定済み候補の1-2決着率。結果待ちは分母から除外。',
        'records':today_rows})
    write_json(HISTORY, {'schema':'boat12-accuracy-history-v2', 'generated_at':now.isoformat(),
        'today':today, 'first_recorded_date': min((r['date'] for r in all_rows), default=None),
        'source':'BOAT12 saved pre-deadline forecasts and BoatraceCSV published results',
        'record_count':len(all_rows), 'days_with_predictions':len([d for d in days if predictions[d]]),
        'coverage_note':'累計は保存済みの実験予測がある日からの累計。予測ログのないレース・日は対象外。',
        'records':all_rows})
    print(f'ACCURACY HISTORY UPDATED {now.isoformat()} days={len(days)} rows={len(all_rows)} confirmed={sum(r["result_status"] == "confirmed" for r in all_rows)} today_status={statuses.get(today)}',flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
