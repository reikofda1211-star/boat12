
import requests

dates = [
    "20211010",
    "20231010",
    "20251010",
    "20261010",
]

for date in dates:
    folder = date[:6]
    short = date[2:]

    print(f"\n【{date}】")

    for label, code in [
        ("公式番組表", "B"),
        ("公式競走成績", "K"),
    ]:
        url = (
            f"https://www1.mbrace.or.jp/od2/"
            f"{code}/{folder}/{code.lower()}{short}.lzh"
        )

        try:
            with requests.get(
                url, timeout=30, stream=True
            ) as response:
                print(
                    label,
                    "取得可能" if response.status_code == 200
                    else f"HTTP {response.status_code}"
                )
        except requests.RequestException as e:
            print(label, "通信エラー:", str(e)[:120])

print("\n確認終了")
