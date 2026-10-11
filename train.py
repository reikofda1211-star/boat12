
import io
import sys
import subprocess
import requests

# LZHを読み取るライブラリを準備
subprocess.check_call([
    sys.executable, "-m", "pip",
    "install", "pylzh"
])

from pylzh import LzhFile

# 2021年10月10日の公式競走成績
url = (
    "https://www1.mbrace.or.jp/od2/"
    "K/202110/k211010.lzh"
)

print("公式データをダウンロード中")

response = requests.get(url, timeout=60)
response.raise_for_status()

print("ダウンロード成功")
print("ファイル容量:", len(response.content), "バイト")

# 圧縮ファイルの中身を確認
with LzhFile(io.BytesIO(response.content), encoding="cp932") as archive:
    names = archive.namelist()

    print("内部ファイル:", names)

    for name in names:
        data = archive.read(name)

        print("ファイル名:", name)
        print("展開後の容量:", len(data), "バイト")

        try:
            text = data.decode("cp932")
        except UnicodeDecodeError:
            text = data.decode("utf-8", errors="replace")

        print("内容の先頭20行:")
        for line in text.splitlines()[:20]:
            print(line)

print("テスト完了")
