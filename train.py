
import csv
import re
import time
import requests
import subprocess
from pathlib import Path
from datetime import date, timedelta, datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

START = date(2021, 10, 12)
END = datetime.now(
    ZoneInfo("Asia/Tokyo")
).date() - timedelta(days=1)

# 1回の取得日数と同時処理数
BATCH_DAYS = 30
WORKERS = 3

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
    temp = DATA_FILE.with_suffix(".tmp")

    with temp.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=FIELDS
        )
        writer.writeheader()

        for key in sorted(records):
            writer.writerow(records[key])

    temp.replace(DATA_FILE)

def fetch_day(day):
    d = day.strftime("%Y%m%d")
    url = (
        "https://www1.mbrace.or.jp/od2/"
        f"K/{d[:6]}/k{d[2:]}.lzh"
    )

    # 通信失敗時は最大3回まで試す
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=45)
            response.raise_for_status()

            # 各日付の圧縮ファイルを直接解凍
            result = subprocess.run(
                ["lha", "-pq2", "-"],
                input=response.content,
                capture_output=True
            )

            # 標準入力非対応環境用の代替処理
            if result.returncode != 0:
                import tempfile
                with tempfile.TemporaryDirectory() as folder:
                    archive = Path(folder) / "result.lzh"
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

            if not rows:
                raise ValueError("抽出結果が0件")

            return d, rows, None

        except Exception as e:
            error = str(e)
            if attempt < 2:
                time.sleep(2 * (attempt + 1))

    return d, [], error

def main():
    records = load_history()
    existing_days = {
        key[0] for key in records
    }

    days = []
    day = START

    while day <= END and len(days) < BATCH_DAYS:
        if day.strftime("%Y%m%d") not in existing_days:
            days.append(day)
        day += timedelta(days=1)

    print("保存済みレース:", len(records), flush=True)
    print("保存済み日数:", len(existing_days), flush=True)
    print("今回取得する日数:", len(days), flush=True)
    print("同時取得数:", WORKERS, flush=True)

    if not days:
        print("対象期間の取得完了")
        return

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

    completed = 0
    failed = []

    with ThreadPoolExecutor(
        max_workers=WORKERS
    ) as pool:
        futures = [
            pool.submit(fetch_day, day)
            for day in days
        ]

        for future in as_completed(futures):
            d, rows, error = future.result()

            if error:
                print(d, "エラー:", error[:120], flush=True)
                failed.append(d)
                continue

            for row in rows:
                key = (
                    row["date"],
                    row["venue"],
                    row["race"]
                )
                records[key] = row

            completed += 1

            # 1日分ずつ保存して途中結果を保護
            save_history(records)

            print(
                d, "取得:", len(rows), "レース",
                flush=True
            )

    hits = sum(
        int(r["hit12"])
        for r in records.values()
    )

    print("\n===== 高速収集結果 =====")
    print("今回取得成功:", completed, "日")
    print("保存済み日数:",
          len({k[0] for k in records}))
    print("保存レース数:", len(records))
    print("1-2決着数:", hits)

    if records:
        print(
            "1-2出現率:",
            f"{hits / len(records) * 100:.2f}%"
        )

    print("取得失敗日:", failed if failed else "なし")
    print("高速収集テスト完了")

if __name__ == "__main__":
    main()
