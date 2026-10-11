
import csv
import io
import requests
from datetime import date

BASE_URL = "https://boatracecsv.github.io/data"

def read_csv(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    text = response.content.decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))

def main():
    today = date.today().strftime("%Y/%m/%d")

    url = f"{BASE_URL}/programs/race_cards/{today}.csv"
    rows = read_csv(url)

    print("取得した出走表:", len(rows))
    print("最初のデータの項目:")
    print(list(rows[0].keys()) if rows else "データなし")

if __name__ == "__main__":
    main()
