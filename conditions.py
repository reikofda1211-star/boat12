"""boat12: weather/exhibition research extension, independent of the v2 site.

Python 3.11+. Existing history/races.csv, training.csv, models and index.html
are never overwritten. Input source: BoatraceCSV public per-source snapshots.
Data specification:
  https://github.com/BoatraceCSV/boatracecsv.github.io/blob/main/docs/data/previews.md
  https://github.com/BoatraceCSV/boatracecsv.github.io/blob/main/docs/data/results.md
Commands:
  python conditions.py --history-days 90 --train   # first historical build
  python conditions.py --history-days 30 --train   # daily incremental build
  python conditions.py                             # today's display only
  python conditions.py --self-test                 # offline integrity checks

A source timestamp is not proof of when that file became publicly available.
Backtests below check the provider's recorded timestamps; actual pre-deadline
site predictions are separately recorded for prospective evaluation.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

VERSION = "conditions-1.0"
JST = ZoneInfo("Asia/Tokyo")
ROOT = Path("history/conditions")
MANIFEST = ROOT / "manifest.json"
REPORT = ROOT / "report.json"
MODEL = ROOT / "model.joblib"
TODAY_OUTPUT = Path("conditions.json")
BASE = "https://raw.githubusercontent.com/BoatraceCSV/boatracecsv.github.io/main/data"
PATHS = {
    "cards": "programs/race_cards", "title": "programs/title",
    "tkz": "previews/tkz", "stt": "previews/stt", "sui": "previews/sui",
    "original": "previews/original_exhibition", "results": "results/realtime",
}
VENUES = ["桐生", "戸田", "江戸川", "平和島", "多摩川", "浜名湖", "蒲郡", "常滑", "津", "三国", "びわこ", "住之江", "尼崎", "鳴門", "丸亀", "児島", "宮島", "徳山", "下関", "若松", "芦屋", "福岡", "唐津", "大村"]
BASE_METRICS = {
    "win": "全国勝率", "rate": "全国2連対率", "local_win": "当地勝率",
    "local_rate": "当地2連対率", "motor_rate": "モーター2連対率", "boat_rate": "ボート2連対率",
}
ORIGINAL_NAMES = {
    "一周": "lap", "一周タイム": "lap", "1周": "lap", "1周タイム": "lap",
    "半周ラップ": "half_lap", "半周": "half_lap",
    "まわり足": "turn", "回り足": "turn", "周り足": "turn",
    "直線": "straight", "直線タイム": "straight",
}
ORIGINAL_LABELS = {"lap": "一周", "half_lap": "半周ラップ", "turn": "まわり足", "straight": "直線"}
MAX_WEATHER_AGE_MIN = 60
MAX_SNAPSHOT_LEAD_MIN = 30
MIN_DAYS = 60
MIN_ROWS = 3000
MIN_VENUE_TRAIN = 80


class RateLimited(RuntimeError):
    pass


def clock() -> datetime:
    return datetime.now(JST)


def normalized(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).strip()


def compact(value: Any) -> str:
    return re.sub(r"\s+", "", normalized(value))


def number(value: Any, lo: float, hi: float) -> float | None:
    try:
        result = float(normalized(value).replace("%", ""))
        return result if math.isfinite(result) and lo <= result <= hi else None
    except (TypeError, ValueError):
        return None


def timestamp(value: Any) -> datetime | None:
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        # Reject ambiguous timestamps rather than assuming their timezone.
        return result.astimezone(JST) if result.tzinfo is not None else None
    except (TypeError, ValueError):
        return None


def deadline(day: date, value: Any) -> datetime | None:
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", normalized(value))
    if not match:
        return None
    try:
        return datetime(day.year, day.month, day.day, int(match[1]), int(match[2]), tzinfo=JST)
    except ValueError:
        return None


def json_load(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(content)
    os.replace(tmp, path)


def atomic_json(path: Path, data: Any) -> None:
    atomic_bytes(path, json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8"))


def record_path(day: date) -> Path:
    return ROOT / "days" / f"{day:%Y%m%d}.jsonl.gz"


def save_records(day: date, rows: list[dict]) -> None:
    data = "".join(json.dumps(r, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n" for r in rows)
    atomic_bytes(record_path(day), gzip.compress(data.encode("utf-8"), mtime=0))


def read_records(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def fetch_csv(kind: str, day: date, session: requests.Session) -> tuple[list[dict], dict]:
    url = f"{BASE}/{PATHS[kind]}/{day:%Y/%m/%d}.csv"
    last_error = ""
    for attempt in range(3):
        try:
            response = session.get(url, timeout=(10, 35), headers={"User-Agent": "boat12-conditions-research/1.0"})
            if response.status_code in (403, 429):
                raise RateLimited(f"{kind}: HTTP {response.status_code}; access stopped, do not bypass")
            if response.status_code == 404:
                return [], {"url": url, "status": "not_found", "rows": 0}
            response.raise_for_status()
            if len(response.content) > 20_000_000:
                raise ValueError("CSV is unexpectedly large")
            try:
                text = response.content.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = response.content.decode("cp932")
            reader = csv.DictReader(io.StringIO(text))
            if not reader.fieldnames or "レースコード" not in reader.fieldnames:
                raise ValueError("CSV header is missing レースコード")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError("duplicate column names")
            rows = list(reader)
            if any(None in r or any(v is None for v in r.values()) for r in rows):
                raise ValueError("CSV has a malformed row")
            return rows, {"url": url, "status": "ok", "rows": len(rows), "sha256": hashlib.sha256(response.content).hexdigest()}
        except RateLimited:
            raise
        except (requests.RequestException, UnicodeError, ValueError, csv.Error) as exc:
            last_error = f"{type(exc).__name__}: {exc}"[:200]
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
        finally:
            # At most two sessions run in parallel; avoid per-race scraping.
            time.sleep(0.25)
    return [], {"url": url, "status": "error", "rows": 0, "error": last_error}


def keyed(rows: list[dict], day: date, audit: Counter) -> dict[str, dict]:
    output: dict[str, dict] = {}
    duplicates: set[str] = set()
    for row in rows:
        code = normalized(row.get("レースコード"))
        if not re.fullmatch(r"\d{12}", code) or not code.startswith(day.strftime("%Y%m%d")):
            audit["wrong_day_or_code"] += 1
            continue
        if not 1 <= int(code[8:10]) <= 24 or not 1 <= int(code[10:12]) <= 12:
            audit["invalid_venue_race"] += 1
            continue
        if normalized(row.get("レース日")) != day.isoformat():
            audit["wrong_date_field"] += 1
            continue
        if code in output and output[code] != row:
            duplicates.add(code)
            audit["conflicting_duplicate"] += 1
        else:
            output[code] = row
    for code in duplicates:
        output.pop(code, None)
    return output


def valid_snapshot(row: dict | None, day: date, cutoff: datetime | None, now: datetime) -> tuple[dict | None, str | None]:
    if not row:
        return None, "not_published"
    snap = timestamp(row.get("取得日時"))
    own_cutoff = deadline(day, row.get("締切時刻"))
    if cutoff is None or snap is None or own_cutoff is None:
        return None, "missing_timestamp"
    if own_cutoff != cutoff:
        return None, "inconsistent_deadline"
    if snap.date() != day or snap >= cutoff or snap > now:
        return None, "not_pre_deadline"
    if cutoff - snap > timedelta(minutes=MAX_SNAPSHOT_LEAD_MIN):
        return None, "snapshot_too_old"
    return row, None


def join_day(day: date, tables: dict[str, list[dict]], now: datetime, historical: bool) -> tuple[list[dict], dict]:
    audit: Counter = Counter()
    maps = {kind: keyed(rows, day, audit) for kind, rows in tables.items()}
    records = []
    for code, card in maps.get("cards", {}).items():
        jo, race = int(code[8:10]), int(code[10:12])
        regs = [normalized(card.get(f"艇{i}_登録番号")) for i in range(1, 7)]
        if not all(re.fullmatch(r"\d{4}", x) for x in regs) or len(set(regs)) != 6:
            audit["incomplete_six_boats"] += 1
            continue
        title = maps.get("title", {}).get(code, {})
        cancelled = any(term in normalized(title.get("中止状態")) + normalized(title.get("電話投票締切予定")) for term in ("中止", "順延"))
        snaps = {kind: maps.get(kind, {}).get(code) for kind in ("sui", "tkz", "stt", "original")}
        cutoffs = {deadline(day, r.get("締切時刻")) for r in snaps.values() if r}
        cutoffs.discard(None)
        title_cutoff = deadline(day, title.get("電話投票締切予定"))
        if title_cutoff:
            cutoffs.add(title_cutoff)
        # Changing or contradictory schedule: withhold, do not guess.
        cutoff = next(iter(cutoffs)) if len(cutoffs) == 1 else None
        valid, issues = {}, []
        for kind, row in snaps.items():
            valid[kind], issue = valid_snapshot(row, day, cutoff, now)
            if issue:
                issues.append(f"{kind}:{issue}")
        if cancelled:
            issues.append("cancelled")
        players = []
        for i in range(1, 7):
            p = {"boat": i, "registration": regs[i - 1], "name": normalized(card.get(f"艇{i}_選手名"))}
            for metric, label in BASE_METRICS.items():
                p[metric] = number(card.get(f"艇{i}_{label}"), 0, 100 if metric.endswith("rate") else 10)
            tkz, stt = valid["tkz"] or {}, valid["stt"] or {}
            p.update({
                "exhibition": number(tkz.get(f"艇{i}_展示タイム"), 4, 12),
                "tilt": number(tkz.get(f"艇{i}_チルト"), -1, 4),
                "exhibition_st": number(stt.get(f"艇{i}_スタート展示"), -1, 2),
                "course": number(stt.get(f"艇{i}_コース"), 1, 6), "original": {},
            })
            players.append(p)
        sui = valid["sui"] or {}
        observed_at = None
        observation = normalized(sui.get("気象観測時刻"))
        if re.fullmatch(r"\d{3,4}", observation):
            observation = observation.zfill(4)
            observed_at = deadline(day, observation[:2] + ":" + observation[2:])
        sui_time = timestamp(sui.get("取得日時"))
        observation_ok = bool(observed_at and cutoff and sui_time and observed_at <= sui_time and cutoff - observed_at <= timedelta(minutes=MAX_WEATHER_AGE_MIN))
        wind_code = number(sui.get("風向"), 0, 16)
        weather = {
            "wind_speed": number(sui.get("風速(m)"), 0, 40),
            "wind_direction_code": str(int(wind_code)) if wind_code is not None and wind_code.is_integer() else None,
            "wave_cm": number(sui.get("波の高さ(cm)"), 0, 200),
            "air_c": number(sui.get("気温(℃)"), -20, 55),
            "water_c": number(sui.get("水温(℃)"), -5, 45),
            "weather_code": normalized(sui.get("天候")) or None,
            "observed_at": observed_at.isoformat() if observed_at else None,
        }
        if not observation_ok:
            issues.append("weather:observation_unconfirmed_or_old")
        # Original values are keyed by the measurement LABEL, never by slot number.
        original = valid["original"]
        original_kinds = []
        original_ok = bool(original)
        if original:
            count = number(original.get("計測数"), 1, 3)
            if count is None or not count.is_integer():
                original_ok = False
            else:
                for slot in range(1, int(count) + 1):
                    kind = ORIGINAL_NAMES.get(compact(original.get(f"計測項目{slot}")))
                    if kind is None or kind in original_kinds:
                        original_ok = False
                        issues.append("original:unknown_or_duplicate_label")
                        continue
                    original_kinds.append(kind)
                    for i, p in enumerate(players, 1):
                        if compact(original.get(f"艇{i}_選手名")) != compact(p["name"]):
                            original_ok = False
                            issues.append("original:name_mismatch")
                        value = number(original.get(f"艇{i}_値{slot}"), 0.1, 90)
                        if value is None:
                            original_ok = False
                        p["original"][kind] = value
        base_ok = all(p[k] is not None for p in players for k in ("win", "rate", "motor_rate"))
        courses = [p["course"] for p in players]
        stt_ok = sorted(x for x in courses if x is not None) == [1, 2, 3, 4, 5, 6] and all(p["exhibition_st"] is not None for p in players)
        basic_ok = bool(base_ok and not cancelled and cutoff and observation_ok and valid["tkz"] and valid["stt"] and stt_ok and all(p["exhibition"] is not None for p in players) and weather["wind_speed"] is not None and weather["wave_cm"] is not None and weather["wind_direction_code"] is not None)
        source_times = {kind: r["取得日時"] for kind, r in valid.items() if r}
        basic_times = [timestamp(source_times.get(k)) for k in ("sui", "tkz", "stt")]
        basic_available = max(basic_times) if all(basic_times) else None
        original_available = max([basic_available, timestamp(source_times.get("original"))]) if basic_available and original_ok else None
        label = None
        result_time = None
        if historical:
            result = maps.get("results", {}).get(code, {})
            first, second = number(result.get("1着_艇番"), 1, 6), number(result.get("2着_艇番"), 1, 6)
            result_time = timestamp(result.get("取得日時"))
            # No result weather, actual-race ST or finishing time is ever used as a feature.
            if first is not None and second is not None and first.is_integer() and second.is_integer() and first != second and cutoff and result_time and result_time > cutoff and result_time <= now:
                label = int(first == 1 and second == 2)
            else:
                audit["no_valid_result"] += 1
        record = {
            "version": VERSION, "code": code, "date": day.isoformat(), "venue": VENUES[jo - 1], "venue_code": f"{jo:02d}", "race": race,
            "cutoff": cutoff.isoformat() if cutoff else None, "cancelled": cancelled,
            "basic_ready": basic_ok, "original_ready": bool(basic_ok and original_ok),
            "original_kinds": sorted(original_kinds), "players": players, "weather": weather,
            "source_times": source_times, "basic_available": basic_available.isoformat() if basic_available else None,
            "original_available": original_available.isoformat() if original_available else None,
            "issues": sorted(set(issues)), "label": label,
            "result_recorded_at": result_time.isoformat() if result_time else None,
        }
        records.append(record)
    records.sort(key=lambda r: r["code"])
    audit.update({"cards": len(maps.get("cards", {})), "records": len(records),
                  "basic_ready": sum(r["basic_ready"] for r in records),
                  "original_ready": sum(r["original_ready"] for r in records),
                  "labelled": sum(r["label"] is not None for r in records)})
    return records, dict(audit)


def get_day(day: date, historical: bool, budget_end: float | None = None) -> tuple[list[dict], dict]:
    tables, sources = {}, {}
    kinds = ["cards", "tkz", "stt", "sui", "original", "results"] if historical else ["cards", "title", "tkz", "stt", "sui", "original"]
    with requests.Session() as session:
        for kind in kinds:
            if budget_end is not None and time.monotonic() > budget_end:
                raise TimeoutError("historical request time budget exceeded; resume next run")
            tables[kind], sources[kind] = fetch_csv(kind, day, session)
    rows, audit = join_day(day, tables, clock(), historical)
    info = {"version": VERSION, "checked_at": clock().isoformat(), "sources": sources, "audit": audit,
            "status": "reviewed" if all(sources[k]["status"] == "ok" for k in ("cards", "tkz", "stt", "sui", "results") if k in sources) else "partial"}
    return rows, info


def collect_history(limit: int) -> dict:
    manifest = json_load(MANIFEST, {"version": VERSION, "days": {}})
    today = clock().date()
    # Separate from the existing 3-year history. Availability is checked per day.
    dates = [today - timedelta(days=i) for i in range(1, 367)]
    unseen, retry = [], []
    for day in dates:
        old = manifest["days"].get(day.isoformat())
        if not old or old.get("version") != VERSION:
            unseen.append(day)
        else:
            checked = timestamp(old.get("checked_at"))
            delay = 1 if old.get("status") != "reviewed" else 7
            if checked and clock() - checked >= timedelta(days=delay):
                retry.append(day)
    # Recheck some partial days without letting a missing date block all progress.
    retry_slots = min(2, len(retry), max(0, limit - 1))
    selected = unseen[:limit - retry_slots] + retry[:retry_slots]
    if len(selected) < limit:
        selected += retry[retry_slots:retry_slots + limit - len(selected)]
    if not selected:
        return manifest
    print(f"直前データ取得対象: {len(selected)}日 (最大2並列)", flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        budget_end = time.monotonic() + 20 * 60
        futures = {pool.submit(get_day, day, True, budget_end): day for day in selected}
        for future in as_completed(futures):
            day = futures[future]
            try:
                rows, info = future.result()
            except RateLimited:
                for pending in futures:
                    pending.cancel()
                raise
            except Exception as exc:
                info = {"version": VERSION, "checked_at": clock().isoformat(), "status": "error", "error": str(exc)[:200]}
                rows = []
            if rows:
                save_records(day, rows)
            manifest["days"][day.isoformat()] = info
            atomic_json(MANIFEST, manifest)
            print(day.isoformat(), info["status"], info.get("audit", {}), flush=True)
    return manifest


def numeric_feature(row: dict, key: str, values: list[float | None], center: bool = False) -> None:
    finite = [x for x in values if x is not None]
    mean = sum(finite) / len(finite) if finite else None
    for i, value in enumerate(values, 1):
        row[f"{key}_b{i}"] = (value - mean if center and value is not None and mean is not None else value)
    if all(x is not None for x in values):
        row[f"{key}_gap12"] = values[0] - values[1]
        rest = sum(values[2:]) / 4
        row[f"{key}_gap1others"] = values[0] - rest
        row[f"{key}_gap2others"] = values[1] - rest
    else:
        row.update({f"{key}_gap12": None, f"{key}_gap1others": None, f"{key}_gap2others": None})


def features(record: dict, scope: str) -> tuple[dict, list[str]]:
    players = record["players"]
    row = {"venue": record["venue"], "race": str(record["race"])}
    categories = ["venue", "race"]
    for key in BASE_METRICS:
        numeric_feature(row, key, [p[key] for p in players])
    if scope == "base":
        return row, categories
    weather = record["weather"]
    for key in ("wind_speed", "wave_cm", "air_c", "water_c"):
        row[key] = weather[key]
    row["wind_direction"] = weather["wind_direction_code"] or "missing"
    row["venue_wind_direction"] = record["venue"] + ":" + row["wind_direction"]
    row["weather_code"] = weather["weather_code"] or "missing"
    categories += ["wind_direction", "venue_wind_direction", "weather_code"]
    # No assumption that a numeric direction code means headwind/tailwind.
    numeric_feature(row, "exhibition", [p["exhibition"] for p in players], center=True)
    numeric_feature(row, "exhibition_st", [p["exhibition_st"] for p in players])
    numeric_feature(row, "tilt", [p["tilt"] for p in players])
    for i, p in enumerate(players, 1):
        row[f"exhibition_course_b{i}"] = str(int(p["course"])) if p["course"] else "missing"
        categories.append(f"exhibition_course_b{i}")
    if scope == "original":
        row["original_layout"] = "+".join(record["original_kinds"])
        categories.append("original_layout")
        # One lap and half lap are NEVER pooled; all time comparisons are within a race.
        for kind in ("lap", "half_lap", "turn", "straight"):
            numeric_feature(row, "original_" + kind, [p["original"].get(kind) for p in players], center=True)
    return row, categories


def frame(records: list[dict], scope: str, columns: list[str] | None = None):
    import numpy as np
    import pandas as pd
    data = [features(r, scope)[0] for r in records]
    result = pd.DataFrame(data, columns=columns)
    return result.replace({None: np.nan})


def fit_model(train_rows: list[dict], cal_rows: list[dict], scope: str) -> dict:
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    x = frame(train_rows, scope)
    _, categorical = features(train_rows[0], scope)
    numerical = [c for c in x.columns if c not in categorical]
    transform = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)), ("scale", StandardScaler())]), numerical),
        ("category", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    model = Pipeline([("features", transform), ("classifier", LogisticRegression(C=0.03, max_iter=3000, solver="lbfgs"))])
    model.fit(x, np.array([r["label"] for r in train_rows]))
    # The probability calibrator sees a later period, not the final test period.
    scores = model.decision_function(frame(cal_rows, scope, list(x.columns))).reshape(-1, 1)
    calibrator = LogisticRegression(C=1000, max_iter=2000)
    calibrator.fit(scores, np.array([r["label"] for r in cal_rows]))
    return {"pipeline": model, "calibrator": calibrator, "columns": list(x.columns), "scope": scope}


def predict(model: dict, rows: list[dict]):
    x = frame(rows, model["scope"], model["columns"])
    scores = model["pipeline"].decision_function(x).reshape(-1, 1)
    return model["calibrator"].predict_proba(scores)[:, 1]


def calibration_table(labels, probs) -> tuple[list[dict], float]:
    import numpy as np
    bins, ece = [], 0.0
    for lo in np.arange(0, 1, 0.1):
        indexes = (probs >= lo) & (probs < lo + 0.1 if lo < 0.9 else probs <= 1)
        n = int(indexes.sum())
        if n:
            mean, rate = float(probs[indexes].mean()), float(labels[indexes].mean())
            bins.append({"lower": round(float(lo), 1), "n": n, "predicted": mean, "observed": rate})
            ece += n / len(labels) * abs(mean - rate)
    return bins, ece


def day_bootstrap_gain(days: list[str], per_row_gain) -> list[float]:
    import numpy as np
    groups = {day: [] for day in sorted(set(days))}
    for day, value in zip(days, per_row_gain):
        groups[day].append(float(value))
    totals = np.array([sum(v) for v in groups.values()])
    counts = np.array([len(v) for v in groups.values()])
    rng = np.random.default_rng(613)
    ix = rng.integers(0, len(totals), size=(1500, len(totals)))
    values = totals[ix].sum(axis=1) / counts[ix].sum(axis=1)
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def train_models() -> dict:
    import joblib
    import numpy as np
    import sklearn
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    all_rows, seen = [], set()
    today = clock().date()
    for path in sorted((ROOT / "days").glob("*.jsonl.gz")):
        for row in read_records(path):
            if row.get("version") != VERSION or row.get("label") not in (0, 1) or row["code"] in seen:
                continue
            if not today - timedelta(days=366) <= date.fromisoformat(row["date"]) < today:
                continue
            seen.add(row["code"])
            all_rows.append(row)
    all_rows.sort(key=lambda r: r["code"])
    digest = hashlib.sha256(json.dumps(all_rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    report = {"version": VERSION, "generated_at": clock().isoformat(), "labelled_records": len(all_rows), "data_sha256": digest,
              "source": "BoatraceCSV per-source pre-deadline timestamps", "scopes": {},
              "warning": "提供元の取得日時に基づく過去検証。ファイルの公開時刻・実際の締切前到達を保証せず、的中率や利益も保証しません。"}
    package = {"version": VERSION, "sklearn_version": sklearn.__version__, "scopes": {}, "report": report}
    for scope in ("normal", "original"):
        key = "basic_ready" if scope == "normal" else "original_ready"
        rows = [r for r in all_rows if r[key]]
        days = sorted({r["date"] for r in rows})
        sub = {"status": "insufficient_data", "rows": len(rows), "days": len(days), "probabilities_enabled": False,
               "period_start": days[0] if days else None, "period_end": days[-1] if days else None}
        report["scopes"][scope] = sub
        if len(days) < MIN_DAYS or len(rows) < MIN_ROWS:
            continue
        cal_start, test_start = days[int(len(days) * 0.6)], days[int(len(days) * 0.8)]
        train = [r for r in rows if r["date"] < cal_start]
        cal = [r for r in rows if cal_start <= r["date"] < test_start]
        test = [r for r in rows if r["date"] >= test_start]
        splits = [train, cal, test]
        if min(len(x) for x in splits) < 300 or any(min(Counter(r["label"] for r in x).get(c, 0) for c in (0, 1)) < 30 for x in splits):
            continue
        base_model, live_model = fit_model(train, cal, "base"), fit_model(train, cal, scope)
        y = np.array([r["label"] for r in test])
        pb, pl = predict(base_model, test), predict(live_model, test)
        constant = np.full(len(y), np.mean([r["label"] for r in train]))
        mb, ml, mc = [float(brier_score_loss(y, p)) for p in (pb, pl, constant)]
        llb, lll = float(log_loss(y, pb)), float(log_loss(y, pl))
        gain_ci = day_bootstrap_gain([r["date"] for r in test], (pb - y) ** 2 - (pl - y) ** 2)
        bins, ece = calibration_table(y, pl)
        # Engineering release rule, NOT a proof of predictive accuracy or profit.
        passed = bool(ml < mb and ml < mc and lll <= llb and gain_ci[0] > 0 and ece <= 0.05)
        sub.update({"status": "experimental_pass" if passed else "held_back", "probabilities_enabled": passed,
                    "train_rows": len(train), "calibration_rows": len(cal), "test_rows": len(test),
                    "train_end": max(r["date"] for r in train), "calibration_start": cal_start, "test_start": test_start,
                    "brier_base": mb, "brier_conditions": ml, "brier_constant": mc,
                    "logloss_base": llb, "logloss_conditions": lll, "auc": float(roc_auc_score(y, pl)),
                    "brier_gain_day_bootstrap_95ci": gain_ci, "calibration_ece": ece, "calibration_bins": bins,
                    "gate": "same test races; improvement over base + constant, day-block CI lower>0, ECE<=0.05"})
        package["scopes"][scope] = {"base": base_model, "conditions": live_model, "report": sub,
                                     "venue_counts": dict(Counter(r["venue"] for r in train)),
                                     "layout_counts": dict(Counter(r["venue"] + ":" + "+".join(r["original_kinds"]) for r in train))}
    ROOT.mkdir(parents=True, exist_ok=True)
    tmp = MODEL.with_suffix(".tmp")
    joblib.dump(package, tmp)
    os.replace(tmp, MODEL)
    atomic_json(REPORT, report)
    print("直前モデル検証:", json.dumps(report, ensure_ascii=False), flush=True)
    return report


def load_models() -> tuple[dict | None, str | None]:
    if not MODEL.exists():
        return None, "条件付きモデルは未作成です。初回は過去データの取得・学習を実行してください。"
    import joblib
    import sklearn
    # Only load this repository's own trusted artifact. Never download arbitrary pickle files.
    package = joblib.load(MODEL)
    if package.get("version") != VERSION or package.get("sklearn_version") != sklearn.__version__:
        return None, "モデルと実行環境の版が違います。再学習するまで確率を表示しません。"
    generated = timestamp(package.get("report", {}).get("generated_at"))
    if not generated or clock() - generated > timedelta(days=8):
        return None, "学習の更新が8日以上確認できません。確率表示を保留しています。"
    return package, None


def active_until(record: dict, scope: str, now: datetime) -> datetime | None:
    cutoff = timestamp(record.get("cutoff"))
    observed = timestamp(record["weather"].get("observed_at"))
    available = timestamp(record.get("original_available" if scope == "original" else "basic_available"))
    if not cutoff or not observed or not available or available > now:
        return None
    kinds = ["sui", "tkz", "stt"] + (["original"] if scope == "original" else [])
    source_stamps = [timestamp(record["source_times"].get(k)) for k in kinds]
    if not all(source_stamps):
        return None
    oldest = min(source_stamps)
    expiry = min(cutoff - timedelta(seconds=30), oldest + timedelta(minutes=20), observed + timedelta(minutes=MAX_WEATHER_AGE_MIN), now + timedelta(minutes=12))
    return expiry if now < expiry else None


def update_today() -> dict:
    now = clock()
    day = now.date()
    try:
        records, info = get_day(day, False)
    except Exception as exc:
        output = {"version": VERSION, "date": day.isoformat(), "generated_at": clock().isoformat(), "status": "download_failed", "message": f"当日データの取得失敗: {type(exc).__name__}", "races": [], "sources": {}}
        atomic_json(TODAY_OUTPUT, output)
        raise
    package, message = None, None
    try:
        package, message = load_models()
    except Exception as exc:
        message = f"モデル読込失敗（{type(exc).__name__}）。データだけ表示しています。"
    now = clock()
    output = {"version": VERSION, "date": day.isoformat(), "generated_at": now.isoformat(), "status": "ok" if records else "waiting",
              "message": message, "sources": info["sources"], "counts": info["audit"], "races": [],
              "report": package["report"] if package else json_load(REPORT, {})}
    prospective = []
    for record in records:
        item = dict(record)
        item.pop("label", None)
        item.pop("result_recorded_at", None)
        item.update({"probability": None, "base_probability": None, "probability_scope": None, "expires_at": None})
        cutoff = timestamp(record.get("cutoff"))
        if record["cancelled"]:
            item["state"] = "中止・順延"
        elif cutoff and cutoff <= now:
            item["state"] = "締切済み"
        elif not record["basic_ready"]:
            item["state"] = "風・展示データ待ち"
        else:
            item["state"] = "検証待ち（データ表示のみ）"
            if package:
                options = ["original", "normal"] if record["original_ready"] else ["normal"]
                for scope in options:
                    model = package["scopes"].get(scope)
                    if not model or not model["report"]["probabilities_enabled"] or model["venue_counts"].get(record["venue"], 0) < MIN_VENUE_TRAIN:
                        continue
                    layout = record["venue"] + ":" + "+".join(record["original_kinds"])
                    if scope == "original" and model["layout_counts"].get(layout, 0) < MIN_VENUE_TRAIN:
                        continue
                    expiry = active_until(record, scope, now)
                    if expiry is None:
                        item["state"] = "古い観測・締切直前のため確率非表示"
                        break
                    p, pb = float(predict(model["conditions"], [record])[0]), float(predict(model["base"], [record])[0])
                    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (p, pb)):
                        continue
                    item.update({"probability": p, "base_probability": pb, "probability_scope": scope,
                                 "expires_at": expiry.isoformat(), "state": "直前推定（実験）"})
                    prospective.append({"code": record["code"], "date": day.isoformat(), "generated_at": now.isoformat(),
                                        "cutoff": record["cutoff"], "input_available": record["original_available" if scope == "original" else "basic_available"],
                                        "probability": p, "base_probability": pb, "scope": scope,
                                        "data_sha256": package["report"]["data_sha256"]})
                    break
        output["races"].append(item)
    output["races"].sort(key=lambda r: (r["probability"] is None, -(r["probability"] or 0), r["code"]))
    output["predicted_races"] = len(prospective)
    atomic_json(TODAY_OUTPUT, output)
    # Record only site-generated predictions made before the known cutoff.
    # Historical reconstructed predictions are never put in this prospective log.
    if prospective:
        folder = ROOT / "predictions" / day.isoformat()
        atomic_json(folder / (now.strftime("%H%M%S") + ".json"), prospective)
    print(f"当日表示: {day} / 出走表 {len(records)} / 直前確率 {len(prospective)}", flush=True)
    return output


def self_test() -> None:
    """Synthetic offline fixtures; never used as training evidence."""
    import copy
    d = date(2026, 5, 6)
    code = "202605061801"
    common = {"レースコード": code, "レース日": d.isoformat(), "締切時刻": "08:38", "取得日時": "2026-05-06T08:30:23+09:00"}
    card = {"レースコード": code, "レース日": d.isoformat()}
    tkz, stt, sui, original, result = [dict(common) for _ in range(5)]
    sui.update({"気象観測時刻": "0805", "風速(m)": "0", "風向": "5", "波の高さ(cm)": "1", "天候": "1", "気温(℃)": "20", "水温(℃)": "24"})
    original.update({"計測数": "2", "計測項目1": "まわり足", "計測項目2": "一周"})
    result.update({"取得日時": "2026-05-06T08:45:00+09:00", "1着_艇番": "1", "2着_艇番": "2", "風速(m)": "39"})
    for i in range(1, 7):
        card.update({f"艇{i}_登録番号": str(4000 + i), f"艇{i}_選手名": f"試験選手{i}"})
        for key, label in BASE_METRICS.items():
            card[f"艇{i}_{label}"] = "30" if key.endswith("rate") else "5"
        tkz.update({f"艇{i}_展示タイム": str(6.8 + i / 100), f"艇{i}_チルト": "0"})
        stt.update({f"艇{i}_コース": str(i), f"艇{i}_スタート展示": "-0.03" if i == 1 else "0.15"})
        original.update({f"艇{i}_選手名": f"試験選手{i}", f"艇{i}_値1": str(11 + i / 10), f"艇{i}_値2": str(37 + i / 10)})
    tables = {"cards": [card], "tkz": [tkz], "stt": [stt], "sui": [sui], "original": [original], "results": [result]}
    now = datetime(2026, 5, 7, 9, tzinfo=JST)
    def get(t):
        return join_day(d, t, now, True)[0][0]
    r = get(tables)
    assert r["basic_ready"] and r["original_ready"] and r["label"] == 1
    assert r["weather"]["wind_speed"] == 0  # Not the post-race wind value 39.
    assert r["players"][0]["exhibition_st"] == -0.03
    assert r["players"][0]["original"]["turn"] == 11.1
    assert r["players"][0]["original"]["lap"] == 37.1
    assert number("nan", 0, 10) is None and number("", 0, 10) is None
    late = copy.deepcopy(tables); late["tkz"][0]["取得日時"] = "2026-05-06T08:39:00+09:00"
    assert not get(late)["basic_ready"]
    old = copy.deepcopy(tables); old["sui"][0]["気象観測時刻"] = "0700"
    assert not get(old)["basic_ready"]
    half = copy.deepcopy(tables); half["original"][0]["計測項目2"] = "半周ラップ"
    rh = get(half)
    assert "half_lap" in rh["players"][0]["original"] and "lap" not in rh["players"][0]["original"]
    absent = copy.deepcopy(tables); absent["original"] = []
    assert get(absent)["basic_ready"] and not get(absent)["original_ready"]
    bad = copy.deepcopy(tables); bad["original"][0]["艇2_選手名"] = "別人"
    assert not get(bad)["original_ready"]
    conflict = copy.deepcopy(tables); other = dict(conflict["tkz"][0]); other["艇1_展示タイム"] = "7.99"; conflict["tkz"].append(other)
    assert not get(conflict)["basic_ready"]
    missing = copy.deepcopy(tables); missing["tkz"][0]["艇6_展示タイム"] = ""
    assert not get(missing)["basic_ready"]
    bad_deadline = copy.deepcopy(tables); bad_deadline["stt"][0]["締切時刻"] = "08:40"
    assert not get(bad_deadline)["basic_ready"]
    race_now = datetime(2026, 5, 6, 8, 33, tzinfo=JST)
    assert active_until(r, "original", race_now) is not None
    assert active_until(r, "original", datetime(2026, 5, 6, 8, 38, tzinfo=JST)) is None
    f, cats = features(r, "original")
    assert "original_lap_b1" in f and "original_half_lap_b1" in f and f["original_half_lap_b1"] is None
    assert abs(sum(f[f"original_lap_b{i}"] for i in range(1, 7))) < 1e-8
    assert not any("label" in key or "result" in key for key in f)
    print("OFFLINE CHECKS PASSED: timestamps, missing data, name match, original labels, leakage, expiry, duplicates, zero wind, negative exhibition ST", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-days", type=int, default=0)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not 0 <= args.history_days <= 90:
        parser.error("--history-days must be between 0 and 90")
    error = None
    if args.history_days:
        try:
            collect_history(args.history_days)
        except Exception as exc:
            error = exc
            print(f"履歴収集失敗: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    if args.train:
        try:
            train_models()
        except Exception as exc:
            error = exc
            # Do not let an old model appear newly validated after a failed run.
            atomic_json(ROOT / "last_training_error.json", {"at": clock().isoformat(), "error": str(exc)[:500]})
            print(f"学習失敗: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    try:
        update_today()
    except Exception as exc:
        error = exc
        print(f"当日更新失敗: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
