
import re
import requests
import subprocess
from pathlib import Path

# 2021年10月10日の公式競走成績
date = "20211010"
url = (
    "https://www1.mbrace.or.jp/od2/"
    "K/202110/k211010.lzh"
)

# LZH解凍ツールをインストール
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

# 公式データを取得
print("公式競走成績を取得中", flush=True)
response = requests.get(url, timeout=60)
response.raise_for_status()

archive = Path("/tmp/boat_result.lzh")
archive.write_bytes(response.content)

# 解凍して日本語テキストにする
result = subprocess.run(
    ["lha", "-pq2", str(archive)],
    capture_output=True,
    check=True
)

text = result.stdout.decode("cp932", errors="replace")

# 2連単の払戻表から結果を抽出
# 3連単→3連複→2連単の順に並んでいる
pattern = re.compile(
    r"^\s*(\d{1,2})R\s+"
    r"([1-6]-[1-6]-[1-6])\s+[\d,]+\s+"
    r"[1-6]-[1-6]-[1-6]\s+[\d,]+\s+"
    r"([1-6]-[1-6])\s+([\d,]+)"
)

total = 0
hits = 0
payouts = []

print("\n取得したレースの一部:")

for line in text.splitlines():
    match = pattern.search(line)
    if not match:
        continue

    race, result3, result12, payout = match.groups()
    payout = int(payout.replace(",", ""))

    total += 1

    if result12 == "1-2":
        hits += 1
        payouts.append(payout)

    if total <= 15:
        print(
            f"{race}R / 2連単 {result12} / "
            f"{payout}円"
        )

print("\n===== 集計結果 =====")
print("対象日:", date)
print("集計できたレース:", total)
print("1-2決着:", hits)

if total:
    print(
        "1-2出現率:",
        f"{hits / total * 100:.2f}%"
    )

if payouts:
    print(
        "1-2平均払戻金:",
        f"{sum(payouts) / len(payouts):.0f}円"
    )

if total < 50:
    print(
        "注意: 抽出数が少ないため、"
        "払戻表の形式を再確認してください"
    )

print("集計テスト完了")
