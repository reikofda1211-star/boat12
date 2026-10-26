import csv
import io
import json
from urllib.request import urlopen
from session_st import compare_session_st

BASE = "https://raw.githubusercontent.com/BoatraceCSV/boatracecsv.github.io/main/data"
DAY = "2026/10/25"

def read_csv(path):
    url = f"{BASE}/{path}/{DAY}.csv"
    with urlopen(url, timeout=30) as response:
        content = response.read(20_000_001)
    if len(content) > 20_000_000:
        raise ValueError("CSV too large")
    return list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))

cards = read_csv("programs/race_cards")
results = read_csv("results/realtime")

result_map = {r["レースコード"]: r for r in results}

print("RESULT_COLUMNS:", list(results[0].keys()) if results else [])

samples = []
for card in cards:
    code = card.get("レースコード")
    result = result_map.get(code)
    if not result:
        continue

    st = compare_session_st(card)
    if st["st_difference"] is None:
        continue

    samples.append({
        "race": code,
        **st,
        "result_raw": result
    })

print(json.dumps({
    "matched_races": len(samples),
    "examples": samples[:3]
}, ensure_ascii=False, indent=2))

# 今節平均ST差と実際の着順を集計
groups = {
    "1号艇が0.05秒以上速い": [],
    "ST差が0.05秒未満": [],
    "2号艇が0.05秒以上速い": []
}

for sample in samples:
    diff = sample["st_difference"]
    if diff <= -0.05:
        key = "1号艇が0.05秒以上速い"
    elif diff >= 0.05:
        key = "2号艇が0.05秒以上速い"
    else:
        key = "ST差が0.05秒未満"

    groups[key].append(sample)

for name, races in groups.items():
    total = len(races)
    hits12 = sum(
        r["result_raw"].get("1着_艇番") == "1"
        and r["result_raw"].get("2着_艇番") == "2"
        for r in races
    )
    hits21 = sum(
        r["result_raw"].get("1着_艇番") == "2"
        and r["result_raw"].get("2着_艇番") == "1"
        for r in races
    )
    print(json.dumps({
        "group": name,
        "races": total,
        "1-2_rate": round(hits12 / total * 100, 1) if total else None,
        "2-1_rate": round(hits21 / total * 100, 1) if total else None
    }, ensure_ascii=False))
