
import csv
import io
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

BASE = "https://boatracecsv.github.io/data"

def read_csv(url):
    response = requests.get(url, timeout=30)
    if response.status_code == 404:
        return []
    response.raise_for_status()
    try:
        text = response.content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = response.content.decode("cp932")
    return list(csv.DictReader(io.StringIO(text)))

def main():
    today = datetime.now(ZoneInfo("Asia/Tokyo")).date()
    total = 0
    hits = 0
    matched_days = 0

    # 昨日から過去7日間をテスト
    for days_ago in range(1, 8):
        day = today - timedelta(days=days_ago)
        path = day.strftime("%Y/%m/%d")

        cards = read_csv(
            f"{BASE}/programs/race_cards/{path}.csv"
        )
        results = read_csv(
            f"{BASE}/results/realtime/{path}.csv"
        )

        result_map = {
            r["レースコード"]: r
            for r in results
            if r.get("レースコード")
        }

        checked = 0
        day_hits = 0

        for card in cards:
            code = card.get("レースコード")
            result = result_map.get(code)
            if not result:
                continue

            first = str(
                result.get("1着_艇番", "")
            ).strip()
            second = str(
                result.get("2着_艇番", "")
            ).strip()

            # 着順不明や不成立は除外
            if not first.isdigit() or not second.isdigit():
                continue

            checked += 1
            if first == "1" and second == "2":
                day_hits += 1

        total += checked
        hits += day_hits

        if checked:
            matched_days += 1

        print(
            f"{day}: 照合 {checked}レース、"
            f"1-2決着 {day_hits}レース"
        )

    print("--------------------")
    print("照合できた日数:", matched_days)
    print("照合できたレース:", total)
    print("1-2決着:", hits)

    if total:
        print(
            "全体の1-2出現率:",
            f"{hits / total * 100:.2f}%"
        )
    else:
        print("照合できるデータがありません")

if __name__ == "__main__":
    main()
