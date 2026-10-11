
import re
import requests
import subprocess
import tempfile
from pathlib import Path

URL = (
    "https://www1.mbrace.or.jp/od2/"
    "B/202310/b231010.lzh"
)

print("2023年の公式番組表を取得中", flush=True)

response = requests.get(URL, timeout=60)
response.raise_for_status()

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

with tempfile.TemporaryDirectory() as folder:
    archive = Path(folder) / "cards.lzh"
    archive.write_bytes(response.content)

    result = subprocess.run(
        ["lha", "-pq2", str(archive)],
        capture_output=True,
        check=True
    )

text = result.stdout.decode("cp932", errors="replace")

# 艇番・選手登録番号・名前・年齢・支部・体重・級別
pattern = re.compile(
    r"^\s*([1-6])\s+"
    r"(\d{4})(.+?)(\d{2})"
    r"([^\d\s]+)(\d{2})"
    r"(A1|A2|B1|B2)\s+"
    r"(.*)$"
)

players = []

for line in text.splitlines():
    match = pattern.match(line)
    if not match:
        continue

    boat, reg, name, age, branch, weight, grade, stats = (
        match.groups()
    )

    # 勝率などの数値を抽出
    numbers = re.findall(r"\d+\.\d+|\d+", stats)

    players.append({
        "boat": boat,
        "reg": reg,
        "name": name.strip(),
        "age": age,
        "branch": branch,
        "weight": weight,
        "grade": grade,
        "numbers": numbers,
        "raw": line.strip()
    })

    if len(players) == 6:
        break

print("\n===== 最初の6選手 =====")
print("抽出人数:", len(players))

for p in players:
    print("\n--------------------")
    print("艇番:", p["boat"])
    print("登録番号:", p["reg"])
    print("選手名:", p["name"])
    print("級別:", p["grade"])
    print("年齢:", p["age"])
    print("支部:", p["branch"])
    print("数値データ:", p["numbers"])
    print("元の行:", p["raw"])

if len(players) != 6:
    print("注意: 6選手を抽出できませんでした")

print("\n抽出テスト完了")
