"""BOAT12: pre-deadline exhibition/weather feature gate for future trifecta model.

This module DOES NOT produce odds or adjusted probabilities. Only validated
pre-deadline observations can be passed to a separately trained/validated model.
"""
from datetime import datetime, timedelta
import math

MAX_WEATHER_MINUTES = 20
MAX_SNAPSHOT_MINUTES = 30

def _time(v):
    if not isinstance(v, str): return None
    try:
        x = datetime.fromisoformat(v.replace('Z', '+00:00'))
        return x if x.tzinfo else None
    except ValueError:
        return None

def _number(v, low, high):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v) if math.isfinite(v) and low <= v <= high else None

def extract_predeadline(record, generated_at=None):
    """Return numerical/categorical dict, or (None, explanation).

    Expects a per-race record from conditions.json or a pre-deadline archived
    conditions snapshot. Never infer that 'data was collected before the cutoff'
    solely from the date of a file or the label of the race.
    """
    if not isinstance(record, dict): return None, 'invalid_record'
    cutoff = _time(record.get('cutoff'))
    generated = _time(generated_at)
    if not cutoff or not generated or generated >= cutoff:
        return None, 'missing_or_late_snapshot_timestamp'
    if cutoff - generated > timedelta(minutes=MAX_SNAPSHOT_MINUTES):
        return None, 'stale_snapshot'
    if record.get('cancelled') is not False:
        return None, 'cancelled_or_unknown'
    if record.get('basic_ready') is not True:
        return None, 'basic_exhibition_not_verified'
    weather = record.get('weather')
    players = record.get('players')
    if not isinstance(weather, dict) or not isinstance(players, list) or len(players)!=6:
        return None, 'missing_features'
    wind = _number(weather.get('wind_speed'),0,40)
    wave = _number(weather.get('wave_cm'),0,150)
    direction = weather.get('wind_direction_code')
    if wind is None or wave is None or not isinstance(direction,str) or not direction.isdigit() or not 0<=int(direction)<=16:
        return None,'missing_weather'
    p = sorted(players,key=lambda x:x.get('boat',99))
    if [x.get('boat') for x in p] != [1,2,3,4,5,6]: return None,'invalid_boats'
    result={'wind_speed':wind,'wave_cm':wave,'wind_direction_code':int(direction)}
    courses=[]
    for x in p:
        i=x['boat']
        for source,key,lo,hi in [
            ('motor_rate','motor_rate',0,100),('boat_rate','boat_rate',0,100),
            ('exhibition','exhibition',4,12),('exhibition_st','exhibition_st',-1,2),
            ('course','exhibition_course',1,6),('tilt','tilt',-3,3)]:
            value=_number(x.get(source),lo,hi)
            if value is None: return None,f'missing_{source}_boat_{i}'
            result[f'b{i}_{key}']=value
        courses.append(int(x['course']))
        if record.get('original_ready') is True:
            original=x.get('original')
            if not isinstance(original,dict): return None,f'missing_original_{i}'
            for k in ('lap','turn','straight'):
                v=_number(original.get(k),0,120)
                if v is None: return None,f'missing_original_{k}_{i}'
                result[f'b{i}_original_{k}']=v
    if sorted(courses)!=[1,2,3,4,5,6]:return None,'invalid_exhibition_courses'
    result['original_available']=bool(record.get('original_ready'))
    return result,None

if __name__=='__main__':
    from datetime import timezone
    now=datetime(2026,10,15,8,tzinfo=timezone.utc)
    race={'cutoff':'2026-10-15T08:10:00+00:00','basic_ready':True,'cancelled':False,
          'original_ready':False,'weather':{'wind_speed':3.0,'wave_cm':5.0,'wind_direction_code':'3'},
          'players':[{'boat':i,'motor_rate':30.0,'boat_rate':32.0,'exhibition':6.8,
                      'exhibition_st':.12,'course':float(i),'tilt':-.5} for i in range(1,7)]}
    features,error=extract_predeadline(race,now.isoformat())
    assert error is None and features['b1_motor_rate']==30
    features,error=extract_predeadline(race,'2026-10-15T08:11:00+00:00')
    assert features is None and error=='missing_or_late_snapshot_timestamp'
    race['basic_ready']=False
    features,error=extract_predeadline(race,now.isoformat())
    assert features is None and error=='basic_exhibition_not_verified'
    print('pre-deadline feature validation tests: PASS')
