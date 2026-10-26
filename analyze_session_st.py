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
