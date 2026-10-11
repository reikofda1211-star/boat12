
import requests
import csv
import io
from datetime import datetime
from zoneinfo import ZoneInfo

# 全国24場の出走データを取得
def main():
    now = datetime.now(ZoneInfo("Asia/Tokyo"))
    date = now.strftime("%Y/%m/%d")

    url = (
        "https://boatracecsv.github.io/"
        f"data/programs/race_cards/{date}.csv"
    )

    print("全国24場 1-2分析システム")
    print("対象日:", date)
    print("出走データを取得中...")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        content = response.content
        try:
            decoded = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            decoded = content.decode("cp932")

        reader = csv.DictReader(io.StringIO(decoded))
        rows = list(reader)

        if not rows:
            print("出走データが空です")
            return

        print("データ取得成功！")
        print("取得行数:", len(rows))
        print("取得項目:", reader.fieldnames)
        print("最初のデータ:", rows[0])

    except requests.RequestException as e:
        print("データ取得エラー:", e)
        raise

if __name__ == "__main__":
    main()
