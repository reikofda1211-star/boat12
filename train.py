
import requests

BASE = "https://boatracecsv.github.io/data"

dates = [
    "2021/10/10",
    "2022/10/10",
    "2023/10/10",
    "2024/10/10",
    "2025/10/10",
    "2026/05/06",
    "2026/10/10",
]

for date in dates:
    print(f"\n【{date}】")

    for name, path in [
        ("出走表", "programs/race_cards"),
        ("結果", "results/realtime"),
    ]:
        url = f"{BASE}/{path}/{date}.csv"

        try:
            response = requests.get(url, timeout=20)
            if response.status_code == 200:
                print(f"{name}: 取得成功")
            elif response.status_code == 404:
                print(f"{name}: データなし")
            else:
                print(f"{name}: HTTP {response.status_code}")
        except requests.RequestException as e:
            print(f"{name}: 通信エラー {e}")

print("\n調査終了")
