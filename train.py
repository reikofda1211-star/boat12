
import re
import requests
import subprocess
import time
from pathlib import Path
from datetime import date, timedelta
from collections import defaultdict

START = date(2021, 10, 1)
DAYS = 30

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

print("解凍ツールを準備中", flush=True)
subprocess.run(
    ["sudo", "apt-get", "update", "-qq"],
    check=True, stdout=subprocess.DEVNULL
)
subprocess.run(
    ["sudo", "apt-get", "install", "-y", "lhasa"],
    check=True, stdout=subprocess.DEVNULL
)

venue_stats = defaultdict(lambda: [0, 0])
race_stats = defaultdict(lambda: [0, 0])
all_total = 0
all_hits = 0
successful_days = 0
failed_days = []

session = requests.Session()

for day_number in range(DAYS):
    day = START + timedelta(days=day_number)
    day_text = day.strftime("%Y%m%d")

    url = (
        "https://www1.mbrace.or.jp/od2/"
        f"K/{day_text[:6]}/k{day_text[2:]}.lzh"
    )

    try:
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
        day_total = 0
        day_hits = 0

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
            race = int(race)
            hit = int(result12 == "1-2")

            venue_stats[venue][0] += 1
            venue_stats[venue][1] += hit

            race_stats[race][0] += 1
            race_stats[race][1] += hit

            day_total += 1
            day_hits += hit

        all_total += day_total
        all_hits += day_hits
        successful_days += 1

        print(
            f"{day_text}: {day_total}レース "
            f"1-2決着 {day_hits}",
            flush=True
        )

        if day_total == 0:
            failed_days.append(day_text + " (抽出0件)")

    except Exception as e:
        failed_days.append(day_text)
        print(
            f"{day_text}: エラー {e}",
            flush=True
        )

    time.sleep(0.2)

print("\n===== 30日間の結果 =====")
print("取得できた日数:", successful_days)
print("集計レース数:", all_total)
print("1-2決着数:", all_hits)

if all_total:
    print(
        "全体の1-2出現率:",
        f"{all_hits / all_total * 100:.2f}%"
    )

print("\n===== 競艇場別 =====")
for venue, (total, hits) in sorted(
    venue_stats.items(),
    key=lambda x: x[1][1] / x[1][0],
    reverse=True
):
    print(
        f"{venue}: {hits}/{total} "
        f"({hits / total * 100:.2f}%)"
    )

print("\n===== レース番号別 =====")
for race, (total, hits) in sorted(race_stats.items()):
    print(
        f"{race}R: {hits}/{total} "
        f"({hits / total * 100:.2f}%)"
    )

print("\n取得・抽出できなかった日:")
print(failed_days if failed_days else "なし")
print("30日間の集計テスト完了")
