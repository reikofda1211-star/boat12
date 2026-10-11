
import re
import requests
import subprocess
from pathlib import Path
from collections import defaultdict

DATES = ["20211010", "20231010", "20251010"]

VENUES = [
    "桐生", "戸田", "江戸川", "平和島", "多摩川",
    "浜名湖", "蒲郡", "常滑", "津", "三国",
    "びわこ", "住之江", "尼崎", "鳴門", "丸亀",
    "児島", "宮島", "徳山", "下関", "若松",
    "芦屋", "福岡", "唐津", "大村"
]

# 解凍ツールを用意
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

# 以前成功した払戻表の抽出式
PATTERN = re.compile(
    r"^\s*(\d{1,2})R\s+"
    r"([1-6]-[1-6]-[1-6])\s+[\d,]+\s+"
    r"[1-6]-[1-6]-[1-6]\s+[\d,]+\s+"
    r"([1-6]-[1-6])\s+([\d,]+)"
)

stats_venue = defaultdict(lambda: [0, 0])
stats_race = defaultdict(lambda: [0, 0])

for date in DATES:
    url = (
        "https://www1.mbrace.or.jp/od2/"
        f"K/{date[:6]}/k{date[2:]}.lzh"
    )

    print(f"\n【{date}】", flush=True)

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    archive = Path("/tmp/boat_result.lzh")
    archive.write_bytes(response.content)

    result = subprocess.run(
        ["lha", "-pq2", str(archive)],
        capture_output=True,
        check=True
    )

    text = result.stdout.decode("cp932", errors="replace")

    venue = None
    counted = 0
    unknown = 0

    for line in text.splitlines():
        # 空白を除いて競艇場を判定
        compact = re.sub(r"\s+", "", line)

        if "ボートレース" in compact:
            after = compact.split("ボートレース", 1)[1]
            matched_venues = [
                v for v in VENUES if after.startswith(v)
            ]
            if matched_venues:
                venue = matched_venues[0]

        match = PATTERN.search(line)
        if not match:
            continue

        race, result3, result12, payout = match.groups()
        race = int(race)

        # 競艇場が不明なレースは場別集計から除外
        if venue is None:
            unknown += 1
            continue

        hit = int(result12 == "1-2")

        stats_venue[venue][0] += 1
        stats_venue[venue][1] += hit

        stats_race[race][0] += 1
        stats_race[race][1] += hit

        counted += 1

    print("集計できたレース:", counted)
    print("競艇場不明:", unknown)

print("\n===== 競艇場別 =====")

for venue, (total, hits) in sorted(
    stats_venue.items(),
    key=lambda item: item[1][1] / item[1][0],
    reverse=True
):
    print(
        f"{venue}: {hits}/{total} "
        f"({hits / total * 100:.2f}%)"
    )

print("\n===== レース番号別 =====")

for race, (total, hits) in sorted(stats_race.items()):
    print(
        f"{race}R: {hits}/{total} "
        f"({hits / total * 100:.2f}%)"
    )

total = sum(v[0] for v in stats_venue.values())
hits = sum(v[1] for v in stats_venue.values())

print("\n===== 全体 =====")
print("集計レース数:", total)
print("1-2決着数:", hits)

if total:
    print("1-2出現率:", f"{hits / total * 100:.2f}%")

print("分析テスト完了")
