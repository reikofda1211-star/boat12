#!/usr/bin/env python3
"""Research-only comparison of pre-race and pre-deadline trifecta models.

Reads already audited data. Does NOT change current published trifecta_model.json.
Both candidates are trained/calibrated/tested on the same chronological partitions.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp
import trifecta_train as base

SOURCE=Path('history/trifecta/live_dataset.json')
REPORT=Path('trifecta_live_model_report.json')
MODEL=Path('history/trifecta/live_model_research.json')
MIN_ROWS=2400
MIN_DAYS=30
MIN_PART=300
F=30


def row_to_base(row):
    f=row['features']
    record={'date':row['day'],'venue':base.VENUES[row['venue']-1],
            'race':str(row['race']),'result123':row['finish']}
    for i in range(1,7):
        # Basic factors below were included in the audited record only if genuinely observed.
        # win/rate/local_* are not present in phase-2 features: they are therefore
        # deliberately omitted from this *matched* experiment rather than fabricated.
        record[f'b{i}_motor_rate']=f[f'b{i}_motor_rate']
        record[f'b{i}_boat_rate']=f[f'b{i}_boat_rate']
    return record


def base_features(rows):
    # Pre-deadline audited rows contain motor and boat rates but not historical
    # nationwide/local win stats. Treat unavailable fields explicitly as absent,
    # both in the baseline and augmented candidate.
    x=np.empty((len(rows),6,6),float)
    for n,r in enumerate(rows):
        f=r['features']
        for i in range(1,7):
            x[n,i-1]=[0,0,0,0,f[f'b{i}_motor_rate'],f[f'b{i}_boat_rate']]
    x-=x.mean(axis=1,keepdims=True)
    x/=base.SCALES
    return x


def live_features(rows):
    arr=np.zeros((len(rows),6,F),float)
    for n,r in enumerate(rows):
        f=r['features']
        wind=f['wind_speed']/10
        wave=f['wave_cm']/20
        direction=int(f['wind_direction_code'])
        if direction<0 or direction>16:raise ValueError('invalid wind direction')
        for i in range(1,7):
            b=i-1
            course=f[f'b{i}_exhibition_course']
            if int(course)!=course or not 1<=course<=6:raise ValueError('invalid course')
            z=arr[n,b]
            z[0]=f[f'b{i}_exhibition']
            z[1]=f[f'b{i}_exhibition_st']
            z[2]=(course-i)/5
            z[3]=(course-3.5)/3
            z[4]=f[f'b{i}_tilt']
            z[5]=wind*(i-3.5)/3
            z[6]=wave*(i-3.5)/3
            z[7]=wind*(course-3.5)/3
            z[8]=wave*(course-3.5)/3
            original=bool(f.get('original_available'))
            z[9]=int(original)
            for j,k in enumerate(('lap','turn','straight'),start=10):
                v=f.get(f'b{i}_original_{k}') if original else None
                if original and (not isinstance(v,(int,float)) or not math.isfinite(v)):
                    raise ValueError('missing claimed original data')
                z[j]=v or 0
            z[13+direction]=(i-3.5)/3
        # Relative comparisons within a race, including original section times.
        for index,scale in [(0,.35),(1,.15),(4,1),(10,3),(11,1),(12,1)]:
            arr[n,:,index]=(arr[n,:,index]-arr[n,:,index].mean())/scale
    if not np.isfinite(arr).all():raise ValueError('nonfinite features')
    return arr


def residual_objective(w, base_z, fx, finish, stage, lam=.015):
    n=len(fx)
    active=np.ones((n,6),bool)
    for prev in range(stage):active[np.arange(n),finish[:,prev]]=False
    z=base_z+np.einsum('nbf,f->nb',fx,w)
    z[~active]=-np.inf
    normal=logsumexp(z,axis=1)
    loss=float(np.mean(normal-z[np.arange(n),finish[:,stage]])+.5*lam*np.dot(w,w))
    e=np.exp(z-normal[:,None]);e[np.arange(n),finish[:,stage]]-=1
    grad=np.einsum('nb,nbf->f',e,fx)/n+lam*w
    return loss,grad


def fit_residual(bw,bx,v,fx,finish):
    weights=[]
    for stage in range(3):
        z=base.stage_logits(bw[stage],bx,v)
        opt=minimize(residual_objective,np.zeros(F),args=(z,fx,finish,stage),
                     jac=True,method='L-BFGS-B',options={'maxiter':180,'ftol':1e-9})
        if not opt.success or not np.isfinite(opt.x).all():
            raise RuntimeError('live residual optimizer failed: '+str(opt.message))
        weights.append(opt.x)
    return np.asarray(weights)


def live_log_prob(bw,rw,bx,v,fx):
    scores=[base.stage_logits(bw[s],bx,v)+np.einsum('nbf,f->nb',fx,rw[s]) for s in range(3)]
    result=np.empty((len(bx),120),float)
    for j,(a,b,c) in enumerate(base.ORDERS):
        rem=[k for k in range(6) if k!=a]
        rem3=[k for k in rem if k!=b]
        result[:,j]=scores[0][:,a]-logsumexp(scores[0],axis=1)+scores[1][:,b]-logsumexp(scores[1][:,rem],axis=1)+scores[2][:,c]-logsumexp(scores[2][:,rem3],axis=1)
    return result


def calibrated(logp, y):
    def loss(logt):
        p=base.temperature_prob(logp,float(np.exp(logt)))
        return float(-np.log(np.clip(p[np.arange(len(y)),y],1e-15,1)).mean())
    opt=minimize_scalar(loss,bounds=(math.log(.5),math.log(4)),method='bounded')
    if not opt.success:raise ValueError('calibration failed')
    t=float(np.exp(opt.x))
    if loss(0)<=loss(opt.x):t=1.
    return t


def run():
    doc=json.loads(SOURCE.read_text(encoding='utf-8'))
    if doc.get('schema')!='boat12-live-dataset-v1':raise ValueError('unexpected input schema')
    seen=set();rows=[]
    for r in doc.get('rows',[]):
        code=r['code']
        if code in seen:raise ValueError('duplicate race')
        seen.add(code)
        assert r['day']==code[:8]
        if r['finish'] not in ['-'.join(map(str,p+1)) for p in base.ORDERS]:raise ValueError('invalid finish')
        rows.append(r)
    rows.sort(key=lambda r:r['code'])
    dates=sorted(set(r['day'] for r in rows))
    report={'status':'insufficient_data','rows':len(rows),'days':len(dates),'generated_at':base.now().isoformat(),
      'published':False,'warning':'研究版。既存サイトの推定確率は変更しません。'}
    if len(rows)<MIN_ROWS or len(dates)<MIN_DAYS:
        REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');return report
    cstart,tstart=dates[int(len(dates)*.70)],dates[int(len(dates)*.85)]
    train=[r for r in rows if r['day']<cstart];cal=[r for r in rows if cstart<=r['day']<tstart];test=[r for r in rows if r['day']>=tstart]
    if min(map(len,(train,cal,test)))<MIN_PART:
        REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');return report
    def arrays(p):
        x=base_features(p);v=np.array([r['venue']-1 for r in p],dtype=int)
        f=np.array([[int(z)-1 for z in r['finish'].split('-')] for r in p],dtype=int)
        y=np.array([base.ORDER_INDEX[tuple(z)] for z in f],dtype=int)
        return x,v,f,y,live_features(p)
    bx,v,f,y,lx=arrays(train)
    bw=base.fit_stages(bx,v,f)
    rw=fit_residual(bw,bx,v,lx,f)
    cx,cv,cf,cy,clx=arrays(cal)
    bt=calibrated(base.joint_log_prob(bw,cx,cv),cy)
    lt=calibrated(live_log_prob(bw,rw,cx,cv,clx),cy)
    tx,tv,tf,ty,tlx=arrays(test)
    pbase=base.temperature_prob(base.joint_log_prob(bw,tx,tv),bt)
    plive=base.temperature_prob(live_log_prob(bw,rw,tx,tv,tlx),lt)
    sb=base.stats(pbase,ty);sl=base.stats(plive,ty)
    improved=sl['log_loss']<sb['log_loss'] and sl['brier_sum']<sb['brier_sum']
    report.update({'status':'research_improved' if improved else 'research_no_improvement',
        'published':False,'improved_on_both_probability_scores':improved,
        'dataset_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'train':{'n':len(train),'from':train[0]['day'],'to':train[-1]['day']},
        'calibration':{'n':len(cal),'from':cal[0]['day'],'to':cal[-1]['day']},
        'test':{'n':len(test),'from':test[0]['day'],'to':test[-1]['day']},
        'baseline_same_rows':sb,'live_same_rows':sl,'baseline_temperature':bt,'live_temperature':lt,
        'scope':'共通の事前情報はモーター・ボート2連対率のみ。直前モデルは風・波・展示・ST・展示進入・オリジナル展示を追加。全国／当地勝率は監査済データにないため両モデルとも未使用。',
        'limitations':'同期間・同一レースの時系列テスト。直前モデルが既存公開済み事前モデルを上回った証明ではなく、将来精度・利益の保証でもない。再検証と実運用時の入力一致検査が必要。'})
    MODEL.parent.mkdir(parents=True,exist_ok=True)
    MODEL.write_text(json.dumps({'research_only':True,'schema':'trifecta-live-research-v1',
        'baseline_weights':bw.tolist(),'live_weights':rw.tolist(),
        'baseline_temperature':bt,'live_temperature':lt,'feature_count':F,
        'ready_for_publication':False},ensure_ascii=False)+'\n',encoding='utf-8')
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report


def self_test():
    rng=np.random.default_rng(2026)
    bx=rng.normal(size=(4,6,6));v=np.array([0,1,2,3]);lx=rng.normal(size=(4,6,F))
    bw=rng.normal(scale=.07,size=(3,base.PARAMS));rw=rng.normal(scale=.07,size=(3,F))
    p=np.exp(live_log_prob(bw,rw,bx,v,lx))
    assert np.allclose(p.sum(axis=1),1,atol=1e-10)
    assert np.all(p>=0)
    print('trifecta live model arithmetic: PASS')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');a=ap.parse_args()
    if a.self_test:self_test()
    else: print(json.dumps(run(),ensure_ascii=False,indent=2))
