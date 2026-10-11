"""Experimental 1-2 race predictor. Requires historical result CSV from train.py.
Historical official B (race card) archives are merged only with same-date K results.
Never use a prediction as a guaranteed probability; see validation report.
"""
import argparse
import csv
import io
import json
import math
import re
import subprocess
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import requests
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline

VENUES = ["桐生", "戸田", "江戸川", "平和島", "多摩川", "浜名湖", "蒲郡", "常滑", "津", "三国", "びわこ", "住之江", "尼崎", "鳴門", "丸亀", "児島", "宮島", "徳山", "下関", "若松", "芦屋", "福岡", "唐津", "大村"]
FEATURE_KINDS = ["win", "rate", "local_win", "local_rate", "motor_no", "motor_rate", "boat_no", "boat_rate"]
FEATURE_NAMES = [f"b{i}_{key}" for i in range(1, 7) for key in FEATURE_KINDS]
KEYS = ["date", "venue", "race", "hit12"]
FIELDS = KEYS + FEATURE_NAMES
DATA = Path("history/races.csv")
TRAINING = Path("history/training.csv")
MODEL = Path("history/model.joblib")
REPORT = Path("history/model_report.json")
PAGE = Path("model-preview.html")
# Compact no-whitespace player lines, e.g. 1 4088野田祥子43長崎47B1 3.59 14.63 ...
PLAYER = re.compile(r"^\s*([1-6])\s+(\d{4})(.+?)(\d{2})([^\d\s]+)(\d{2})(A1|A2|B1|B2)\s+(.+)$")
RACE = re.compile(r"^\s*(\d{1,2})\s*[ＲR]\s+\S")
NUM = re.compile(r"\d+(?:\.\d+)?")


def csv_rows(file):
    if not file.exists():
        return []
    with file.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def atomic_csv(file, rows):
    file.parent.mkdir(parents=True, exist_ok=True)
    temp = file.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(file)


def extract(payload):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "day.lzh"
        path.write_bytes(payload)
        cmd = subprocess.run(["lha", "-pq2", str(path)], capture_output=True, check=True, timeout=35)
        return cmd.stdout.decode("cp932", errors="replace")


def parse_cards(text, day):
    venue = None
    race = None
    players = {}
    parsed = {}
    for line in text.splitlines():
        compact = re.sub(r"\s+", "", line)
        if "ボートレース" in compact:
            after = compact.split("ボートレース", 1)[1]
            found = next((v for v in VENUES if after.startswith(v)), None)
            if found:
                venue = found
                race = None
                players = {}
                continue
        rm = RACE.match(line)
        if rm and venue:
            race = str(int(rm.group(1)))
            players = {}
            continue
        pm = PLAYER.match(line)
        if not pm or venue is None or race is None:
            continue
        boat = int(pm.group(1))
        if boat in players:
            continue
        values = NUM.findall(pm.group(8))
        if len(values) < 8:
            continue
        players[boat] = dict(zip(FEATURE_KINDS, values[:8]))
        if len(players) == 6 and set(players) == set(range(1, 7)):
            row = {"date": day, "venue": venue, "race": race}
            for i in range(1, 7):
                for kind in FEATURE_KINDS:
                    row[f"b{i}_{kind}"] = players[i][kind]
            parsed[(day, venue, race)] = row
    return parsed


def collect(max_days):
    labels = {}
    for row in csv_rows(DATA):
        if row.get("hit12") not in ("0", "1"):
            continue
        try:
            key = (row["date"], row["venue"], str(int(row["race"])))
            labels[key] = row["hit12"]
        except (KeyError, ValueError):
            continue
    if not labels:
        raise RuntimeError("history/races.csv に学習対象レースがありません")
    old = {}
    for row in csv_rows(TRAINING):
        if row.get("hit12") in ("0", "1"):
            old[(row["date"], row["venue"], row["race"])] = row
    completed = {k[0] for k in old}
    dates = sorted({k[0] for k in labels} - completed)[:max_days]
    print(f"履歴結果: {len(labels)}件 / 学習済み: {len(old)}件 / 新規対象: {len(dates)}日", flush=True)
    session = requests.Session()
    for day in dates:
        url = f"https://www1.mbrace.or.jp/od2/B/{day[:6]}/b{day[2:]}.lzh"
        try:
            response = session.get(url, timeout=45)
            response.raise_for_status()
            cards = parse_cards(extract(response.content), day)
            day_labels = {k for k in labels if k[0] == day}
            joined = 0
            for key, record in cards.items():
                if key in labels:
                    record["hit12"] = labels[key]
                    old[key] = record
                    joined += 1
            coverage = joined / max(1, len(day_labels))
            print(f"{day}: 番組表={len(cards)} 結果={len(day_labels)} 照合={joined} 一致率={coverage:.1%}", flush=True)
            if coverage < .75:
                # Do not quietly accept broken parsing as a reliable dataset.
                for key in [k for k in old if k[0] == day]:
                    old.pop(key, None)
                print("  警告: 照合率75%未満。この日は保存せず解析方法の確認が必要", flush=True)
            else:
                atomic_csv(TRAINING, [old[k] for k in sorted(old)])
        except (requests.RequestException, subprocess.SubprocessError, ValueError) as ex:
            print(f"{day}: 取得失敗: {str(ex)[:130]}", flush=True)
        time.sleep(.4)
    return list(old.values())


def features(row):
    arr = []
    for name in FEATURE_NAMES:
        value = row.get(name, "")
        try:
            arr.append(float(value))
        except (ValueError, TypeError):
            arr.append(math.nan)
    return arr


def train(rows):
    days = sorted(set(row["date"] for row in rows))
    if len(days) < 20 or len(rows) < 1200:
        print(f"学習を保留：現在{len(days)}日・{len(rows)}件。目安20日・1200件以上", flush=True)
        return None
    split = days[int(len(days)*.8)]
    train_rows = [r for r in rows if r["date"] < split]
    test_rows = [r for r in rows if r["date"] >= split]
    y_train = [int(r["hit12"]) for r in train_rows]
    y_test = [int(r["hit12"]) for r in test_rows]
    if len(set(y_train)) < 2 or len(set(y_test)) < 2:
        print("学習保留：的中・不的中の両方が必要", flush=True)
        return None
    pipe = make_pipeline(SimpleImputer(strategy="median"), LogisticRegression(max_iter=2000))
    pipe.fit([features(r) for r in train_rows], y_train)
    preds = pipe.predict_proba([features(r) for r in test_rows])[:,1]
    baseline = sum(y_train)/len(y_train)
    score = brier_score_loss(y_test, preds)
    base_score = brier_score_loss(y_test, [baseline]*len(y_test))
    report = {
        "training_rows": len(train_rows), "validation_rows":len(test_rows),
        "validation_start":split,"train_hit_rate":round(baseline,4),
        "brier_model":round(score,5),"brier_baseline":round(base_score,5),
        "auc":round(roc_auc_score(y_test, preds),4),
        "validated_improvement":bool(score < base_score),
        "warning":"過去データを用いた暫定モデル。確率の較正と取得データの検査が必要。"
    }
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    joblib.dump({"pipeline":pipe,"features":FEATURE_NAMES,"report":report},MODEL)
    print("時系列検証:",json.dumps(report,ensure_ascii=False),flush=True)
    return pipe, report


def preview(model, report):
    # This page is experimental; does not overwrite the live index.html.
    now = datetime.now(ZoneInfo("Asia/Tokyo"))
    address = "https://boatracecsv.github.io/data/programs/race_cards/"+now.strftime("%Y/%m/%d")+".csv"
    try:
        response = requests.get(address, timeout=30)
        response.raise_for_status()
        cards = list(csv.DictReader(io.StringIO(response.content.decode("utf-8-sig"))))
    except Exception as exc:
        print(f"今日の出走表未公開・予測プレビュー保留: {exc}",flush=True)
        return
    mapping = {
        "win":"全国勝率", "rate":"全国2連対率", "local_win":"当地勝率",
        "local_rate":"当地2連対率", "motor_no":"モーター番号",
        "motor_rate":"モーター2連対率", "boat_no":"ボート番号", "boat_rate":"ボート2連対率"
    }
    ready = []
    for card in cards:
        if any(not card.get(f"艇{i}_登録番号") for i in range(1,7)):
            continue
        datum = {}
        for i in range(1,7):
            for key,name in mapping.items():
                datum[f"b{i}_{key}"] = card.get(f"艇{i}_{name}","")
        code = card.get("レース場コード", "")
        try:
            venue = VENUES[int(code)-1]
        except (ValueError, IndexError):
            venue = str(code)
        ready.append((card, venue, datum))
    if not ready:
        print("有効な出走表がありません",flush=True)
        return
    from html import escape
    prob = model.predict_proba([features(d) for _,_,d in ready])[:,1]
    ranked = sorted(zip(ready,prob),key=lambda x:-x[1])
    cards_html = []
    for ((card,venue,_),p) in ranked:
        name1 = escape(card.get("艇1_選手名", ""))
        name2 = escape(card.get("艇2_選手名", ""))
        race = escape(str(card.get("レース回","")))
        cards_html.append(f'<article><strong>{escape(venue)} {race}</strong><span>{p*100:.1f}%</span><small>1号艇 {name1}／2号艇 {name2}</small></article>')
    status = "一定の改善あり" if report["validated_improvement"] else "単純な基準予測より悪い（要改善）"
    text = '''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>1-2確率 実験版</title><style>body{background:#0b1220;color:white;font-family:system-ui;padding:18px;max-width:900px;margin:auto}article{padding:16px;background:#1e293b;border-radius:12px;margin:10px 0;display:flex;gap:10px;justify-content:space-between;flex-wrap:wrap}span{font-size:22px;color:#4ade80;font-weight:bold}small{width:100%;color:#aab8ca}p{line-height:1.7;color:#bcc7d6}</style><h2>全国1-2予測（実験版）</h2>'''+f'<p>対象日：{now:%Y-%m-%d}。過去の時系列検証: {status}。参考推定値であり的中保証はありません。検証Brier: {report["brier_model"]}（基準 {report["brier_baseline"]}）</p>'+''.join(cards_html)+'</html>'
    PAGE.write_text(text,encoding="utf-8")
    print("model-preview.html を作成",flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days",type=int,default=30)
    args = parser.parse_args()
    if not 1 <= args.days <= 60:
        raise SystemExit("--days は1〜60で指定")
    rows=collect(args.days)
    result=train(rows)
    if result:
        preview(*result)

if __name__ == "__main__":
    main()
