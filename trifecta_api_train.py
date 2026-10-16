#!/usr/bin/env python3
"""Open API compatible, no-tilt basic trifecta model; isolated research artifact."""
import argparse
from pathlib import Path
import numpy as np
import trifecta_live_train as research

REPORT=Path('trifecta_api_model_report.json')
MODEL=Path('history/trifecta/api_model_research.json')

def features(rows):
    x=np.zeros((len(rows),6,30),float)
    for n,row in enumerate(rows):
        f=row['features'];wind=float(f['wind_speed'])/10;wave=float(f['wave_cm'])/20
        direction=int(f['wind_direction_code'])
        if not 0<=direction<=16:raise ValueError('invalid wind direction')
        for i in range(1,7):
            z=x[n,i-1];c=float(f[f'b{i}_exhibition_course'])
            if not (c.is_integer() and 1<=c<=6):raise ValueError('invalid course')
            z[0]=float(f[f'b{i}_exhibition']);z[1]=float(f[f'b{i}_exhibition_st'])
            z[2]=(c-i)/5;z[3]=(c-3.5)/3
            # z[4] (tilt) deliberately absent from both train and inference.
            z[5]=wind*(i-3.5)/3;z[6]=wave*(i-3.5)/3
            z[7]=wind*(c-3.5)/3;z[8]=wave*(c-3.5)/3
            z[13+direction]=(i-3.5)/3
        for j,scale in ((0,.35),(1,.15)):
            x[n,:,j]=(x[n,:,j]-x[n,:,j].mean())/scale
    if not np.isfinite(x).all():raise ValueError('nonfinite features')
    return x

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
    if args.self_test:
        f={'wind_speed':4.,'wave_cm':3.,'wind_direction_code':4}
        for i in range(1,7): f.update({f'b{i}_exhibition':6.7+i*.02,f'b{i}_exhibition_st':.1,f'b{i}_exhibition_course':i})
        a=features([{'features':f}]);assert a.shape==(1,6,30) and np.isfinite(a).all() and (a[:,:,4]==0).all()
        print('API-compatible features: PASS');return
    research.live_features=features;research.REPORT=REPORT;research.MODEL=MODEL
    report=research.run(); print('API_MODEL',report.get('status'),report.get('rows'),flush=True)
    # Deliberately leaves research_only and ready_for_publication=false.
if __name__=='__main__': main()
