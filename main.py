
import requests
import csv
import io
import html
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

VENUES = [
    "桐生", "戸田", "江戸川", "平和島", "多摩川", "浜名湖",
    "蒲郡", "常滑", "津", "三国", "びわこ", "住之江",
    "尼崎", "鳴門", "丸亀", "児島", "宮島", "徳山",
    "下関", "若松", "芦屋", "福岡", "唐津", "大村"
]

def number(value, default=0):
    try:
        return float(str(value).replace("%", "").strip())
    except (ValueError, TypeError):
        return default

def get_data():
    now = datetime.now(ZoneInfo("Asia/Tokyo"))
    date = now.strftime("%Y/%m/%d")
    url = (
        "https://boatracecsv.github.io/"
        f"data/programs/race_cards/{date}.csv"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    try:
        content = response.content.decode("utf-8-sig")
    except UnicodeDecodeError:
        content = response.content.decode("cp932")
    rows = list(csv.DictReader(io.StringIO(content)))
    if not rows:
        raise ValueError("出走データが空です")
    return rows, now.strftime("%Y-%m-%d")

def calculate(row):
    racers = []
    for i in range(1, 7):
        prefix = f"艇{i}_"
        win = number(row.get(prefix + "全国勝率"))
        rate = number(row.get(prefix + "全国2連対率"))
        motor = number(row.get(prefix + "モーター2連対率"))
        local = number(row.get(prefix + "当地勝率"))
        racers.append({
            "name": row.get(prefix + "選手名", ""),
            "win": win,
            "rate": rate,
            "motor": motor,
            "local": local
        })

    if any(not row.get(f"艇{i}_登録番号") for i in range(1, 7)):
        return None

    def strength(r):
        return (
            r["win"] * 7
            + r["rate"] * 0.4
            + r["motor"] * 0.2
            + r["local"] * 2
        )

    strengths = [strength(r) for r in racers]
    first = strengths[0]
    second = strengths[1]
    others = strengths[2:]

    # 1号艇の強さと2号艇の相対的な強さを評価
    score = (
        first * 0.55
        + second * 0.35
        - max(others) * 0.20
        - sum(others) / 4 * 0.10
    )

    venue_code = str(row.get("レース場コード", "")).zfill(2)
    try:
        venue_index = int(venue_code) - 1
        venue = VENUES[venue_index]
    except (ValueError, IndexError):
        venue = venue_code

    return {
        "venue": venue,
        "race": row.get("レース回", ""),
        "score": round(score, 2),
        "first": racers[0]["name"],
        "second": racers[1]["name"]
    }

def make_html(results, date):
    cards = []
    for rank, r in enumerate(results, 1):
        cards.append(f"""
        <div class="card">
          <div class="top">
            <span>#{rank}　{html.escape(r['venue'])}
              {html.escape(r['race'])}</span>
            <strong>{r['score']:.1f} pt</strong>
          </div>
          <div class="names">
            1号艇：{html.escape(r['first'])}<br>
            2号艇：{html.escape(r['second'])}
          </div>
        </div>
        """)

    page = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>全国ボート 1-2分析</title>
<style>
body {{
 background:#0b1220;color:#f1f5f9;
 font-family:-apple-system,BlinkMacSystemFont,sans-serif;
 margin:0;padding:18px;
}}
h1 {{font-size:23px}}
.note {{color:#94a3b8;font-size:13px;line-height:1.7}}
.card {{
 background:#1e293b;border-radius:13px;
 padding:17px;margin:12px 0;
}}
.top {{
 display:flex;justify-content:space-between;
 gap:10px;font-weight:bold;
}}
strong {{color:#4ade80;white-space:nowrap}}
.names {{color:#cbd5e1;margin-top:12px;line-height:1.8}}
</style>
</head>
<body>
<h1>全国ボート 1-2ランキング</h1>
<p>対象日：{date}</p>
<p class="note">
取得した出走表を参考スコア順に表示。
スコアは的中確率ではありません。
過去成績による検証は未実施です。
</p>
{''.join(cards)}
</body>
</html>"""
    Path("index.html").write_text(page, encoding="utf-8")

def main():
    rows, date = get_data()
    results = []
    for row in rows:
        result = calculate(row)
        if result is not None:
            results.append(result)

    results.sort(key=lambda x: x["score"], reverse=True)
    make_html(results, date)

    print("出走データ取得数:", len(rows))
    print("ランキング対象数:", len(results))
    print("ランキング上位10件")
    for rank, r in enumerate(results[:10], 1):
        print(
            rank, r["venue"], r["race"],
            "スコア:", r["score"]
        )
    print("index.html を生成しました")

if __name__ == "__main__":
    main()
