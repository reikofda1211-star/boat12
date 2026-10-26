import csv
import io
import json
from datetime import datetime
from urllib.request import urlopen
from zoneinfo import ZoneInfo

from session_st import compare_session_st

day = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y/%m/%d")
url = f"https://boatracecsv.github.io/data/programs/race_cards/{day}.csv"

with urlopen(url, timeout=30) as response:
    content = response.read(8_000_001)
    if len(content) > 8_000_000:
        raise ValueError("CSV too large")

rows = list(csv.DictReader(io.StringIO(content.decode("utf-8-sig"))))

if not rows:
    raise ValueError("出走表データがありません")

results = []
for row in rows:
    result = compare_session_st(row)
    results.append({
        "race": row.get("レースコード"),
        **result
    })

available = sum(
    r["st_difference"] is not None
    for r in results
)

print(json.dumps({
    "total_races": len(results),
    "both_st_available": available,
    "examples": results[:10]
}, ensure_ascii=False, indent=2))
