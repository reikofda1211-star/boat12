#!/usr/bin/env python3
"""BOAT12 trifecta research model, v1.0.

Additive: NEVER changes the existing 1-2 model, training.csv or conditions.py.
Fits three masked choice models to historical ordered finishes. Exports plain
JSON for browser inference. No LLM/API key, hand-assigned probabilities or odds.
A time-ordered fit/calibration/test split is used; test labels are never fitted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import tempfile
import time
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import requests
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp

VERSION = 'boat12-trifecta-1.0'
JST = ZoneInfo('Asia/Tokyo')
VENUES = ['桐生','戸田','江戸川','平和島','多摩川','浜名湖','蒲郡','常滑','津','三国','びわこ','住之江','尼崎','鳴門','丸亀','児島','宮島','徳山','下関','若松','芦屋','福岡','唐津','大村']
METRICS = ['win','rate','local_win','local_rate','motor_rate','boat_rate']
SCALES = np.array([2.,20.,2.,20.,20.,20.])
ORDERS = np.array(list(itertools.permutations(range(6), 3)), dtype=int)
ORDER_INDEX = {tuple(x): i for i, x in enumerate(ORDERS)}
LABEL_PATTERN = re.compile(r'^\s*(\d{1,2})R\s+([1-6]-[1-6]-[1-6])\s+[\d,]+\s+([1-6]-[1-6]-[1-6])\s+[\d,]+\s+([1-6]-[1-6])\s+[\d,]+(?:\s|$)')
INPUT = Path('history/training.csv')
CACHE = Path('history/trifecta/labels')
OUTPUT = Path('trifecta_model.json')
PARAMS = 192  # 6 lane + 24*6 venue/lane + 6 shared + 6*6 lane-specific features
ALPHA = 0.0005
MIN_ROWS, MIN_DAYS, MIN_PARTITION = 2400, 30, 300
MAX_VALIDATION_AGE = 180


def now():
    return datetime.now(JST)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def valid_day(raw):
    s = str(raw)
    if not re.fullmatch(r'\d{8}', s):
        raise ValueError('Date must be YYYYMMDD')
    parsed = datetime.strptime(s, '%Y%m%d').date()
    if parsed.strftime('%Y%m%d') != s:
        raise ValueError('Invalid date')
    return s


def key_of(row):
    day = valid_day(row['date'])
    venue = str(row['venue'])
    race = int(row['race'])
    if venue not in VENUES or str(race) != str(row['race']).lstrip('0') or not 1 <= race <= 12:
        raise ValueError('Invalid venue/race')
    return day, venue, str(race)


def numbers(row):
    values = np.empty((6, 6), dtype=float)
    for b in range(6):
        for m, metric in enumerate(METRICS):
            raw = row.get(f'b{b+1}_{metric}')
            if isinstance(raw, bool):
                raise ValueError('Boolean feature')
            v = float(raw)
            upper = 10 if metric in ('win', 'local_win') else 100
            if not math.isfinite(v) or not 0 <= v <= upper:
                raise ValueError('Missing or out-of-range feature')
            values[b, m] = v
    return (values - values.mean(axis=0, keepdims=True)) / SCALES


def load_training(path=INPUT):
    if not path.exists():
        raise ValueError('history/training.csv がありません。既存の学習用データを先に作成してください。')
    found, duplicate, skipped = {}, set(), Counter()
    with path.open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            try:
                k = key_of(r)
                if k[0] >= now().strftime('%Y%m%d'):
                    skipped['not_past'] += 1
                    continue
                if r.get('hit12') not in ('0', '1'):
                    raise ValueError('Invalid label')
                numbers(r)
                if k in found:
                    duplicate.add(k)
                else:
                    found[k] = r
            except (ValueError, TypeError, KeyError):
                skipped['invalid_row'] += 1
    for k in duplicate:
        found.pop(k, None)
    skipped['duplicate_keys_removed'] = len(duplicate)
    return found, dict(skipped)


def parse_labels(text, day):
    """Parse the existing K archive's payout summary, cross-checking two bets.

    Multiple/ambiguous outcomes for the same race are excluded, not guessed.
    A unique, valid first-three finish is the model's conditional outcome space.
    """
    valid_day(day)
    venue, records, duplicate = None, {}, set()
    for raw in text.splitlines():
        line = unicodedata.normalize('NFKC', raw)
        compact = re.sub(r'\s+', '', line)
        if 'ボートレース' in compact:
            tail = compact.split('ボートレース', 1)[1]
            venue = next((v for v in VENUES if tail.startswith(v)), None)
        match = LABEL_PATTERN.match(line)
        if not match or venue is None:
            continue
        race_s, triple, unordered, exacta = match.groups()
        race = int(race_s)
        boats = triple.split('-')
        if not 1 <= race <= 12 or len(set(boats)) != 3:
            continue
        if set(boats) != set(unordered.split('-')) or exacta != '-'.join(boats[:2]):
            continue
        code = f'{day}{VENUES.index(venue)+1:02d}{race:02d}'
        if code in records:
            duplicate.add(code)
        else:
            records[code] = triple
    for code in duplicate:
        records.pop(code, None)
    return records


def cached_labels(day):
    raw = read_json(CACHE / (day + '.json'))
    if not isinstance(raw, dict) or raw.get('schema') != VERSION or raw.get('day') != day:
        return {}
    result = raw.get('labels')
    if not isinstance(result, dict):
        return {}
    good = {}
    for code, value in result.items():
        if not isinstance(code, str) or not re.fullmatch(day + r'\d{4}', code):
            continue
        if not 1 <= int(code[8:10]) <= 24 or not 1 <= int(code[10:]) <= 12:
            continue
        if isinstance(value, str) and re.fullmatch(r'[1-6]-[1-6]-[1-6]', value) and len(set(value.split('-'))) == 3:
            good[code] = value
    return good


def collect_labels(training, max_days):
    days = sorted({k[0] for k in training}, reverse=True)
    pending = [d for d in days if not cached_labels(d)][:max_days]
    print('結果の追加取得日数:', len(pending), flush=True)
    with requests.Session() as session:
        session.headers['User-Agent'] = 'boat12-trifecta-research/1.0'
        for day in pending:
            url = f'https://www1.mbrace.or.jp/od2/K/{day[:6]}/k{day[2:]}.lzh'
            try:
                # A denied/rate-limited source stops the run. No alternate route.
                with session.get(url, timeout=(10, 35), stream=True) as response:
                    if response.status_code in (401, 403, 429):
                        raise PermissionError(f'取得元がアクセスを制限しました ({response.status_code})。自動再試行しません。')
                    response.raise_for_status()
                    chunks, size = [], 0
                    for part in response.iter_content(65536):
                        size += len(part)
                        if size > 8_000_000:
                            raise ValueError('Archive too large')
                        chunks.append(part)
                    payload = b''.join(chunks)
                with tempfile.TemporaryDirectory() as folder:
                    archive = Path(folder) / 'day.lzh'
                    archive.write_bytes(payload)
                    proc = subprocess.run(['lha', '-pq2', str(archive)], check=True, capture_output=True, timeout=35)
                if len(proc.stdout) > 25_000_000:
                    raise ValueError('Extracted archive too large')
                labels = parse_labels(proc.stdout.decode('cp932', errors='strict'), day)
                expected = {f'{d}{VENUES.index(v)+1:02d}{int(r):02d}' for d,v,r in training if d == day}
                covered = len(set(labels) & expected)
                if not labels or covered / max(1, len(expected)) < .75:
                    raise ValueError('出走表と結果の照合率が75%未満。保存せず解析確認が必要です。')
                write_json(CACHE / (day + '.json'), {'schema':VERSION, 'day':day, 'source':url,
                           'sha256':hashlib.sha256(payload).hexdigest(), 'obtained_at':now().isoformat(), 'labels':labels})
                print(day, '保存結果:', len(labels), '照合:', covered, flush=True)
            except PermissionError:
                raise
            except (requests.RequestException, OSError, ValueError, subprocess.SubprocessError) as exc:
                print(day, '取得保留:', str(exc)[:160], flush=True)
            time.sleep(1)


def joined_rows(training):
    byday = {d:cached_labels(d) for d in {k[0] for k in training}}
    joined, excluded = [], Counter()
    for (day, venue, race), r in sorted(training.items()):
        code = f'{day}{VENUES.index(venue)+1:02d}{int(race):02d}'
        label = byday[day].get(code)
        if not label:
            excluded['missing_result123'] += 1
            continue
        if str(int(label.startswith('1-2-'))) != r['hit12']:
            excluded['old_label_mismatch'] += 1
            continue
        out = dict(r)
        out['result123'] = label
        joined.append(out)
    return joined, dict(excluded)


def stage_logits(w, x, venue):
    w = np.asarray(w, dtype=float)
    return w[:6][None, :] + w[6:150].reshape(24,6)[venue] + np.einsum('nbm,m->nb', x, w[150:156]) + np.einsum('nbm,bm->nb', x, w[156:].reshape(6,6))


def stage_objective(w, x, venue, targets, mask, alpha=ALPHA):
    z = stage_logits(w, x, venue)
    z[~mask] = -np.inf
    n = len(x)
    norm = logsumexp(z, axis=1)
    loss = float(np.mean(norm - z[np.arange(n), targets]) + .5 * alpha * np.dot(w, w))
    residual = np.exp(z - norm[:, None])
    residual[np.arange(n), targets] -= 1
    residual /= n
    grad = np.zeros_like(w)
    grad[:6] = residual.sum(axis=0)
    venue_grad = np.zeros((24,6))
    np.add.at(venue_grad, venue, residual)
    grad[6:150] = venue_grad.ravel()
    grad[150:156] = np.einsum('nb,nbm->m', residual, x)
    grad[156:] = np.einsum('nb,nbm->bm', residual, x).ravel()
    return loss, grad + alpha * w


def fit_stages(x, venue, finish):
    weights = []
    for stage in range(3):
        mask = np.ones((len(x),6), dtype=bool)
        for prev in range(stage):
            mask[np.arange(len(x)), finish[:, prev]] = False
        opt = minimize(stage_objective, np.zeros(PARAMS), args=(x,venue,finish[:,stage],mask),
                       method='L-BFGS-B', jac=True, options={'maxiter':500,'ftol':1e-10,'gtol':1e-6})
        if not opt.success or not np.all(np.isfinite(opt.x)):
            raise ValueError(f'{stage+1}着モデルの最適化が収束しませんでした: {opt.message}')
        weights.append(opt.x)
    return np.asarray(weights)


def joint_log_prob(weights, x, venue):
    scores = [stage_logits(w, x, venue) for w in weights]
    n = len(x)
    answer = np.empty((n,120))
    first_normalizer = logsumexp(scores[0], axis=1)
    for j,(a,b,c) in enumerate(ORDERS):
        remaining2 = [i for i in range(6) if i != a]
        remaining3 = [i for i in range(6) if i not in (a,b)]
        answer[:,j] = scores[0][:,a] - first_normalizer + scores[1][:,b] - logsumexp(scores[1][:,remaining2], axis=1) + scores[2][:,c] - logsumexp(scores[2][:,remaining3], axis=1)
    return answer


def temperature_prob(log_prob, temperature):
    if not math.isfinite(temperature) or not .5 <= temperature <= 4:
        raise ValueError('Invalid temperature')
    z = log_prob / temperature
    return np.exp(z - logsumexp(z, axis=1, keepdims=True))


def chronological_parts(rows):
    days = sorted({r['date'] for r in rows})
    if len(days) < MIN_DAYS or len(rows) < MIN_ROWS:
        return None
    cal_start, test_start = days[int(len(days)*.70)], days[int(len(days)*.85)]
    parts = [[r for r in rows if r['date'] < cal_start],
             [r for r in rows if cal_start <= r['date'] < test_start],
             [r for r in rows if r['date'] >= test_start]]
    if any(len(p) < MIN_PARTITION for p in parts):
        return None
    return parts


def arrays(rows):
    x = np.asarray([numbers(r) for r in rows])
    venue = np.asarray([VENUES.index(r['venue']) for r in rows], dtype=int)
    finish = np.asarray([[int(b)-1 for b in r['result123'].split('-')] for r in rows], dtype=int)
    y = np.asarray([ORDER_INDEX[tuple(f)] for f in finish])
    return x, venue, finish, y


def stats(p, target):
    rows = np.arange(len(p))
    ranked = np.argsort(-p, axis=1, kind='stable')[:,:5]
    hits = np.any(ranked == target[:,None], axis=1)
    coverage = np.take_along_axis(p, ranked, axis=1).sum(axis=1)
    n, rate, z = len(p), float(hits.mean()), 1.95996398454
    centre = (rate+z*z/(2*n))/(1+z*z/n)
    half = z * math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
    bins = []
    for lo in (0,.2,.4,.6,.8):
        mask = (coverage >= lo) & ((coverage < lo+.2) if lo < .8 else (coverage <= 1.0000001))
        count = int(mask.sum())
        if count:
            bins.append({'from':lo,'to':lo+.2,'count':count,'mean_estimate':float(coverage[mask].mean()),'observed_rate':float(hits[mask].mean())})
    return {'n':n, 'log_loss':float(-np.log(np.maximum(p[rows,target],1e-300)).mean()),
            'brier_sum':float(np.mean(np.square(p).sum(axis=1)-2*p[rows,target]+1)),
            'top5_hit_rate':rate, 'top5_mean_estimate':float(coverage.mean()),
            'top5_wilson95':[max(0.,centre-half),min(1.,centre+half)], 'top5_bins':bins}


def fit_model(rows, excluded):
    stamp = now().isoformat()
    out = {'schema':VERSION,'ready':False,'generated_at':stamp,'status':'insufficient_data',
           'message':f'3連単の学習データを収集中：{len(rows)}件。最低{MIN_ROWS}件・{MIN_DAYS}日と各分割300件が必要です。',
           'metrics':METRICS,'scales':SCALES.tolist(),'rows':len(rows),'excluded':excluded,
           'scope':'事前成績のみ。風・展示・進入・オッズは未使用。1〜3着の順序が一意に確定する有効決着を条件とした推定。',
           'warning':'実験的な推定値。実際の的中率や利益の保証ではありません。'}
    parts = chronological_parts(rows)
    if parts is None:
        return out
    train, cal, test = parts
    x,v,finish,y = arrays(train)
    weights = fit_stages(x,v,finish)
    cx,cv,_,cy = arrays(cal)
    cal_log = joint_log_prob(weights,cx,cv)
    def calibrate(log_t):
        p = temperature_prob(cal_log, float(np.exp(log_t)))
        return float(-np.log(np.maximum(p[np.arange(len(cy)),cy],1e-300)).mean())
    fitted = minimize_scalar(calibrate, bounds=(math.log(.5),math.log(4.)), method='bounded')
    if not fitted.success:
        raise ValueError('確率補正が収束しませんでした。')
    temp = float(np.exp(fitted.x))
    if calibrate(0.) <= fitted.fun:
        temp = 1.
    tx,tv,_,ty = arrays(test)
    p = temperature_prob(joint_log_prob(weights,tx,tv), temp)
    # Baseline is estimated on the SAME train partition, never the test period.
    global_counts = np.bincount(y, minlength=120).astype(float)+.5
    global_p = global_counts/global_counts.sum()
    venue_p = np.empty((24,120))
    for venue in range(24):
        count = np.bincount(y[v==venue], minlength=120).astype(float)
        venue_p[venue] = (count+50*global_p)/(count.sum()+50)
    model_stats, base_stats = stats(p,ty), stats(venue_p[tv],ty)
    train_count, test_count = Counter(v.tolist()), Counter(tv.tolist())
    allowed = [int(i)+1 for i in range(24) if train_count[i]>=100 and test_count[i]>=30]
    evidence_end = max(r['date'] for r in test)
    age = (now().date()-datetime.strptime(evidence_end,'%Y%m%d').date()).days
    # Project publication guardrails, not a claim of statistically proven superiority.
    better = model_stats['log_loss'] < base_stats['log_loss'] and model_stats['brier_sum'] < base_stats['brier_sum']
    release = bool(better and age <= MAX_VALIDATION_AGE and allowed)
    out.update({'status':'experimental_ready' if release else 'validation_hold','ready':release,
        'message':'時系列テストで基準を上回った実験モデルです。将来の精度・利益は保証しません。' if release else '検証基準未達・データの古さ・会場別件数不足のため確率を非表示にしています。',
        'weights':weights.tolist(),'temperature':temp,'allowed_venues':allowed,
        'evidence_end':evidence_end,'max_evidence_age_days':MAX_VALIDATION_AGE,
        'report':{'train':{'n':len(train),'from':min(r['date'] for r in train),'to':max(r['date'] for r in train)},
                  'calibration':{'n':len(cal),'from':min(r['date'] for r in cal),'to':max(r['date'] for r in cal)},
                  'test':{'n':len(test),'from':min(r['date'] for r in test),'to':evidence_end},
                  'model':model_stats,'venue_frequency_baseline':base_stats,
                  'better_on_two_scores':bool(better),'venue_training_counts':{str(i+1):train_count[i] for i in range(24)},
                  'venue_test_counts':{str(i+1):test_count[i] for i in range(24)},
                  'limitations':'単一の時系列分割による検証。統計的有意差や全国各場での精度保証ではない。反復開発で同じテストを使い続けない。'}})
    out['model_id'] = hashlib.sha256(json.dumps({'weights':out['weights'],'temperature':temp,'evidence_end':evidence_end},sort_keys=True).encode()).hexdigest()[:16]
    return out


def self_test():
    import unittest
    class Tests(unittest.TestCase):
        def test_all_120(self):
            self.assertEqual(len(ORDERS),120)
            self.assertEqual(len(ORDER_INDEX),120)
            self.assertTrue(all(len(set(x))==3 for x in ORDERS))
        def test_probability_mass(self):
            rng=np.random.default_rng(9)
            x=rng.normal(size=(3,6,6));v=np.array([0,10,23])
            w=rng.normal(scale=.2,size=(3,PARAMS))
            raw=joint_log_prob(w,x,v)
            self.assertTrue(np.allclose(np.exp(raw).sum(axis=1),1,atol=1e-12))
            for t in (.5,1.,4.):
                p=temperature_prob(raw,t)
                self.assertTrue(np.all(p>=0))
                self.assertTrue(np.allclose(p.sum(axis=1),1,atol=1e-12))
                self.assertTrue(np.all(np.sort(p)[:,-5:].sum(axis=1)<1))
        def test_equal_scores(self):
            p=np.exp(joint_log_prob(np.zeros((3,PARAMS)),np.zeros((1,6,6)),np.array([0])))
            self.assertTrue(np.allclose(p,1/120))
        def test_gradient(self):
            rng=np.random.default_rng(8);x=rng.normal(size=(12,6,6));v=np.arange(12)%24
            w=rng.normal(size=PARAMS)*.03;target=np.arange(12)%5+1
            mask=np.ones((12,6),bool);mask[:,0]=False
            _,grad=stage_objective(w,x,v,target,mask)
            for j in (0,5,6,51,149,150,155,156,191):
                plus=w.copy();minus=w.copy();plus[j]+=1e-6;minus[j]-=1e-6
                fd=(stage_objective(plus,x,v,target,mask)[0]-stage_objective(minus,x,v,target,mask)[0])/2e-6
                self.assertAlmostEqual(grad[j],fd,places=7)
        def test_parser_cross_checks(self):
            s='ボートレース徳 山\n 1R 1-2-3 1,250 1-2-3 200 1-2 500\n 2R 2-1-3 800 1-2-3 100 1-2 200\n'
            got=parse_labels(s,'20261013')
            self.assertEqual(got,{'202610131801':'1-2-3'})
            self.assertEqual(parse_labels(s+' 1R 1-3-2 250 1-2-3 200 1-3 100\n','20261013'),{})
        def test_bad_features(self):
            r={f'b{i}_{m}':'2' for i in range(1,7) for m in METRICS}
            self.assertEqual(numbers(r).shape,(6,6))
            for bad in ('NaN','inf','','-1','101',True,None):
                broken=dict(r);broken['b1_win']=bad
                with self.assertRaises((ValueError,TypeError)):
                    numbers(broken)
        def test_time_split(self):
            rows=[{'date':f'202609{d:02d}'} for d in range(1,31) for _ in range(100)]
            a,b,c=chronological_parts(rows)
            self.assertLess(max(r['date'] for r in a),min(r['date'] for r in b))
            self.assertLess(max(r['date'] for r in b),min(r['date'] for r in c))
        def test_small_data_holds(self):
            self.assertFalse(fit_model([], {})['ready'])
        def test_metrics(self):
            p=np.full((10,120),1/120);s=stats(p,np.zeros(10,dtype=int))
            self.assertAlmostEqual(s['top5_mean_estimate'],5/120)
            self.assertAlmostEqual(s['brier_sum'],119/120)
        def test_fit_on_synthetic(self):
            rng=np.random.default_rng(12);x=rng.normal(size=(160,6,6));v=np.arange(160)%4
            finish=np.array([rng.choice(6,3,replace=False) for _ in range(160)])
            w=fit_stages(x,v,finish)
            self.assertEqual(w.shape,(3,PARAMS))
            self.assertTrue(np.all(np.isfinite(w)))
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-days',type=int,default=30)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test();return
    if not 0<=args.history_days<=60:
        raise SystemExit('--history-days は0〜60')
    try:
        training,excluded=load_training()
        if args.history_days:
            collect_labels(training,args.history_days)
        rows,join_excluded=joined_rows(training)
        excluded.update(join_excluded)
        model=fit_model(rows,excluded)
        write_json(OUTPUT,model)
        print(model['status'],model['message'],flush=True)
        if 'report' in model:
            print(json.dumps(model['report'],ensure_ascii=False,indent=2),flush=True)
    except Exception as exc:
        # Fail closed; an old, ready model must not silently survive an error.
        write_json(OUTPUT,{'schema':VERSION,'ready':False,'status':'pipeline_error','generated_at':now().isoformat(),
                          'message':'学習処理が失敗しました。Actionsのログを確認してください。','error_type':type(exc).__name__})
        raise

if __name__=='__main__':
    main()
