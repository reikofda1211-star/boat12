
import csv
import re
import requests
import subprocess
import time
from pathlib import Path
from datetime import date, timedelta, datetime
from zoneinfo import ZoneInfo
from collections import defaultdict

# 取得する期間
START = date(2021, 10, 12)
END = datetime.now(
    ZoneInfo("Asia/Tokyo")
).date() - timedelta(days=1)

# 1回の実行で取得する最大日数
BATCH_DAYS = 7

DATA_FILE = Path("history/races.csv")

VENUES = [
    "桐生", "戸田", "江戸川", "平和島",
    "多摩川", "浜名湖", "蒲郡", "常滑",
    "津", "三国", "びわこ", "住之江",
    "尼崎", "鳴門", "丸亀", "児島",
    "宮島", "徳山", "下関", "若松",
    "芦屋", "福岡", "唐津", "大村"
]

PATTERN = re.compile(
    r"^\s*(\d{1,2})R\s+"
    r"([1-6]-[1-6]-[1-6])\s+[\d,]+\s+"
    r"[1-6]-[1-6]-[1-6]\s+[\d,]+\s+"
    r"([1-6]-[1-6])\s+([\d,]+)"
)

FIELDS = [
    "date", "venue", "race",
    "result12", "payout", "hit12"
]

def load_history():
    records = {}
    if DATA_FILE.exists():
        with DATA_FILE.open(
            encoding="utf-8", newline=""
        ) as f:
            for row in csv.DictReader(f):
                key = (
                    row["date"],
                    row["venue"],
                    row["race"]
                )
                records[key] = row
    return records

def save_history(records):
    DATA_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    with DATA_FILE.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=FIELDS
        )
        writer.writeheader()
        for key in sorted(records):
            writer.writerow(records[key])

def get_results(day, session):
    d = day.strftime("%Y%m%d")
    url = (
        "https://www1.mbrace.or.jp/od2/"
        f"K/{d[:6]}/k{d[2:]}.lzh"
    )

    response = session.get(url, timeout=60)
    response.raise_for_status()

    archive = Path("/tmp/boat_result.lzh")
    archive.write_bytes(response.content)

    result = subprocess.run(
        ["lha", "-pq2", str(archive)],
        capture_output=True,
        check=True
    )

    text = result.stdout.decode(
        "cp932", errors="replace"
    )

    venue = None
    rows = []

    for line in text.splitlines():
        compact = re.sub(r"\s+", "", line)

        if "ボートレース" in compact:
            after = compact.split(
                "ボートレース", 1
            )[1]

            for v in VENUES:
                if after.startswith(v):
                    venue = v
                    break

        match = PATTERN.search(line)
        if not match or venue is None:
            continue

        race, _, result12, payout = match.groups()

        rows.append({
            "date": d,
            "venue": venue,
            "race": str(int(race)),
            "result12": result12,
            "payout": payout.replace(",", ""),
            "hit12": str(int(result12 == "1-2"))
        })

    return rows

def main():
    records = load_history()

    existing_days = {
        key[0] for key in records
    }

    print("保存済みレース:", len(records))

    days = []
    day = START

    while day <= END:
        if day.strftime("%Y%m%d") not in existing_days:
            days.append(day)
        day += timedelta(days=1)

    days = days[:BATCH_DAYS]

    if not days:
        print("対象期間の取得は完了しています")
        return

    print("今回取得する日数:", len(days))

    subprocess.run(
        ["sudo", "apt-get", "update", "-qq"],
        check=True,
        stdout=subprocess.DEVNULL
    )
    subprocess.run(
        ["sudo", "apt-get", "install", "-y", "lhasa"],
        check=True,
        stdout=subprocess.DEVNULL
    )

    session = requests.Session()

    for day in days:
        try:
            rows = get_results(day, session)

            if not rows:
                print(day, "抽出0件・未保存")
                continue

            for row in rows:
                key = (
                    row["date"],
                    row["venue"],
                    row["race"]
                )
                records[key] = row

            print(
                day,
                "取得:", len(rows), "レース",
                flush=True
            )

        except Exception as e:
            print(
                day, "エラー:",
                str(e)[:160],
                flush=True
            )

        time.sleep(0.2)

    save_history(records)

    hits = sum(
        int(row["hit12"])
        for row in records.values()
    )

    print("\n===== 保存結果 =====")
    print("保存レース数:", len(records))
    print(
        "保存済み日数:",
        len({k[0] for k in records})
    )
    print("1-2決着数:", hits)

    if records:
        print(
            "1-2出現率:",
            f"{hits / len(records) * 100:.2f}%"
        )

    print("history/races.csv を作成しました")
    print("保存処理完了")

if __name__ == "__main__":
    main()
