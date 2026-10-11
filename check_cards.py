
import requests
import subprocess
import tempfile
from pathlib import Path

# 2023年10月10日の公式番組表
url = (
    "https://www1.mbrace.or.jp/od2/"
    "B/202310/b231010.lzh"
)

print("公式番組表を取得中", flush=True)

response = requests.get(url, timeout=60)
response.raise_for_status()

print("ダウンロード成功")
print("容量:", len(response.content), "バイト")

# LZH解凍ツールを準備
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
    archive = Path(folder) / "program.lzh"
    archive.write_bytes(response.content)

    result = subprocess.run(
        ["lha", "-pq2", str(archive)],
        capture_output=True,
        check=True
    )

    text = result.stdout.decode(
        "cp932", errors="replace"
    )

lines = text.splitlines()

print("解凍成功！")
print("総行数:", len(lines))
print("===== 番組表の先頭80行 =====")

for i, line in enumerate(lines[:80], 1):
    print(f"{i:03d}: {line}")

print("===== 確認完了 =====")
