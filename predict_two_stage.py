"""Research-only two-stage 1-2 prediction preview. Does not touch production."""
import csv
import io
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import joblib
import pandas as pd
import requests
from compare_two_stage import load
from compare_competitor_features import competitor_features, make_pipeline, EXTRA
from model_v2 import CATEGORICAL, NUMERICAL, METRICS, VENUES, feature_row

OUT = Path("two-stage-preview.csv")
MAP = {"win":"全国勝率","rate":"全国2連対率","local_win":"当地勝率",
       "local_rate":"当地2連対率","motor_rate":"モーター2連対率","boat_rate":"ボート2連対率"}

def main():
    records = load()
    if len(records) < 1200:
        raise SystemExit("Not enough historical records")
    # Training uses only races before the forecast date.
    today = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d")
    records = [r for r in records if r[0].replace("-", "") < today]
    x = []
    y_first, y_12 = [], []
    for _, row, first, hit in records:
        enriched = dict(row)
        enriched.update(competitor_features(enriched))
        x.append(enriched)
        y_first.append(first)
        y_12.append(hit)
    import numpy as np
    x = pd.DataFrame(x, columns=CATEGORICAL + NUMERICAL + EXTRA)
    y_first = np.array(y_first)
    y_12 = np.array(y_12)
    if len(set(y_first)) < 2 or len(set(y_12[y_first == 1])) < 2:
        raise SystemExit("Insufficient class variation")
    direct = make_pipeline(NUMERICAL).fit(x[CATEGORICAL + NUMERICAL], y_12)
    first = make_pipeline(NUMERICAL + EXTRA).fit(x, y_first)
    second = make_pipeline(NUMERICAL + EXTRA).fit(x.loc[y_first == 1], y_12[y_first == 1])
    day = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d")
    url = "https://raw.githubusercontent.com/BoatraceCSV/boatracecsv.github.io/main/data/programs/race_cards/" + day.replace("-", "/") + ".csv"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    try:
        content = response.content.decode("utf-8-sig")
    except UnicodeDecodeError:
        content = response.content.decode("cp932")
    ready = []
    for card in csv.DictReader(io.StringIO(content)):
        if any(not card.get(f"艇{i}_登録番号") for i in range(1,7)):
            continue
        try:
            venue = VENUES[int(card["レース場コード"])-1]
            race = str(int(str(card["レース回"]).replace("R", "")))
        except (ValueError, KeyError, IndexError):
            continue
        raw = {"venue":venue,"race":race}
        for i in range(1,7):
            for metric, col in MAP.items():
                raw[f"b{i}_{metric}"] = card.get(f"艇{i}_{col}", "")
        row = feature_row(raw)
        row.update(competitor_features(row))
        ready.append((venue, race, row))
    if not ready:
        raise SystemExit("No valid race cards; preview not generated")
    frame = pd.DataFrame([r[2] for r in ready], columns=CATEGORICAL + NUMERICAL + EXTRA)
    p_direct = direct.predict_proba(frame[CATEGORICAL + NUMERICAL])[:,1]
    p_first = first.predict_proba(frame)[:,1]
    p_second = second.predict_proba(frame)[:,1]
    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["date","venue","race","direct_12_pct","two_stage_12_pct","first_win_pct","second_given_first_pct"])
        for (venue,race,_),a,b,c in zip(ready,p_direct,p_first,p_second):
            writer.writerow([day,venue,race,round(a*100,2),round(b*c*100,2),round(b*100,2),round(c*100,2)])
    print(f"Created {OUT}: {len(ready)} races (research only)")

if __name__ == "__main__":
    main()
