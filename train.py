
import requests
import subprocess
from pathlib import Path

# 2021年10月10日の公式競走成績
url = (
    "https://www1.mbrace.or.jp/od2/"
    "K/202110/k211010.lzh"
)

print("LZH解凍ツールを準備中", flush=True)

subprocess.run(
    ["sudo", "apt-get", "update", "-qq"],
    check=True
)
subprocess.run(
    ["sudo", "apt-get", "install", "-y", "lhasa"],
    check=True
)

print("公式データをダウンロード中", flush=True)

response = requests.get(url, timeout=60)
response.raise_for_status()

file = Path("/tmp/boat_result.lzh")
file.write_bytes(response.content)

print("取得成功:", len(response.content), "バイト")
print("ファイル内の一覧:")

subprocess.run(
    ["lha", "-l", str(file)],
    check=True
)

print("圧縮ファイルの検査:")

subprocess.run(
    ["lha", "-t", str(file)],
    check=True
)

print("テスト完了")
