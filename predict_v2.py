"""Experimental 1-2 preview, separate from the live index.html."""
import csv
import io
import json
from datetime import datetime
from html import escape
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd
import requests
from model_v2 import feature_row, CATEGORICAL, NUMERICAL, VENUES

PAGE = Path('v2-preview.html')
REPORT = Path('history/model_v2_report.json')
MODEL = Path('history/model_v2.joblib')
MAPPING = {'win':'全国勝率','rate':'全国2連対率','local_win':'当地勝率',
           'local_rate':'当地2連対率','motor_rate':'モーター2連対率','boat_rate':'ボート2連対率'}
CSS = '''body{background:#0b1220;color:#f1f5f9;font:16px system-ui;margin:0;padding:18px}
main{max-width:720px;margin:auto}h1{font-size:24px}.note{color:#b2bed2;line-height:1.7;font-size:14px}
article{background:#1e293b;margin:12px 0;padding:16px;border-radius:14px}
.head{display:flex;justify-content:space-between;gap:12px;font-weight:700}.pct{font-size:25px;color:#86efac}
.names{color:#d6ddeb;margin-top:10px;line-height:1.7}.detail{font-size:13px;color:#aab7cc;margin-top:8px}
a{color:#a5d2ff}'''

def publish(day, report, message, rankings):
    header = f'''<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>1-2予測 実験版</title>
<style>{CSS}</style></head><body><main><h1>全国1-2推定確率（実験版）</h1>
<p class="note">対象日：{escape(day)}<br>未検証の将来予測です。的中・利益は保証されません。<br>
過去検証Brier：{escape(str(report.get('brier_model','未確認')))}／単純な基準：{escape(str(report.get('brier_baseline','未確認')))}</p>
<p class="note">出走表の列対応・年度差・確率較正は検証途中です。参考数値として表示しています。</p>'''
    parts = [header]
    if message:
        parts.append(f'<p class="note">{escape(message)}</p>')
    for rank, (venue, race, one, two, pct, reason) in enumerate(rankings,1):
        parts.append(f'''<article><div class="head"><span>#{rank} {escape(venue)} {escape(race)}R</span>
<span class="pct">{pct*100:.1f}%</span></div><div class="names">1号艇：{escape(one)}<br>2号艇：{escape(two)}</div>
<div class="detail">参考データ：{escape(reason)}</div></article>''')
    parts.append('''<p class="note">参考データは予測への寄与度を表すものではありません。</p>
<p><a href="./">従来ランキングに戻る</a></p></main></body></html>''')
    PAGE.write_text(''.join(parts),encoding='utf-8')
    print('生成:', PAGE,'対象日:',day,'件数:',len(rankings),flush=True)

def main():
    day = datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y-%m-%d')
    report = json.loads(REPORT.read_text(encoding='utf-8')) if REPORT.exists() else {}
    if not report.get('improved_over_baseline') or not MODEL.exists():
        publish(day, report, 'モデルの精度検証が完了していないため、予測を表示していません。', [])
        return
    
    url = 'https://raw.githubusercontent.com/BoatraceCSV/boatracecsv.github.io/main/data/programs/race_cards/'+day.replace('-','/')+'.csv'

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        try:
            content = response.content.decode('utf-8-sig')
        except UnicodeDecodeError:
            content = response.content.decode('cp932')
        rows = list(csv.DictReader(io.StringIO(content)))
    except (requests.RequestException, UnicodeError, csv.Error) as exc:
        publish(day, report, '当日の出走表が取得できません。古い日付の予測は表示していません。', [])
        print('取得失敗:',exc,flush=True)
        return
    ready = []
    for row in rows:
        if any(not row.get(f'艇{i}_登録番号') for i in range(1,7)):
            continue
        try:
            venue = VENUES[int(row['レース場コード'])-1]
            race = str(int(str(row['レース回']).replace('R','')))
        except (ValueError, KeyError, IndexError):
            continue
        fields = {'venue':venue,'race':race}
        for boat in range(1,7):
            for key, col in MAPPING.items():
                fields[f'b{boat}_{key}'] = row.get(f'艇{boat}_{col}', '')
        ready.append((row,venue,race,fields))
    if not ready:
        publish(day,report,'有効な出走表がありません。',[])
        return
    package = joblib.load(MODEL)  # trusted model generated in this repository only
    frame = pd.DataFrame([feature_row(d) for _,_,_,d in ready],columns=CATEGORICAL+NUMERICAL)
    probabilities = package['pipeline'].predict_proba(frame)[:,1]
    ranking = []
    for (row, venue, race, d), p in zip(ready,probabilities):
        refs = []
        for boat in (1,2):
            for label, col in [('全国2連対率','全国2連対率'),('モーター2連対率','モーター2連対率')]:
                value = row.get(f'艇{boat}_{col}','')
                if value:
                    refs.append(f'{boat}号艇{label} {value}')
        ranking.append((venue,race,row.get('艇1_選手名',''),row.get('艇2_選手名',''),float(p),'／'.join(refs)))
    ranking.sort(key=lambda r:r[4],reverse=True)
    publish(day,report,'',ranking)

if __name__ == '__main__':
    main()
