
import requests
import subprocess
from pathlib import Path

# 2021年10月10日の公式競走成績
url = (
    "https://www1.mbrace.or.jp/od2/"
    "K/202110/k211010.lzh"
)

# 解凍ツールを準備
subprocess.run(
    ["sudo", "apt-get", "update", "-qq"],
    check=True
)
subprocess.run(
    ["sudo", "apt-get", "install", "-y", "lhasa"],
    check=True
)

# ダウンロード
print("公式データを取得中", flush=True)
response = requests.get(url, timeout=60)
response.raise_for_status()

archive = Path("/tmp/boat_result.lzh")
archive.write_bytes(response.content)

# LZHの中にあるテキストを取り出す
result = subprocess.run(
    ["lha", "-pq2", str(archive)],
    capture_output=True,
    check=True
)

text = result.stdout.decode("cp932", errors="replace")
lines = text.splitlines()

print("解凍成功！")
print("テキストの行数:", len(lines))
print("----- 内容の先頭60行 -----")

for i, line in enumerate(lines[:60], 1):
    print(f"{i:03d}: {line}")

print("----- 表示終了 -----")
print("テスト完了")
