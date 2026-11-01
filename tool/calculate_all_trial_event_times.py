import csv,json
from pathlib import Path
import numpy as np
B=Path(__file__).resolve().parents[1]/'Data/1029_D1';O=B/'report_aperture_examples'/'all_trials'
rows=[];issues=[]
def crossing(t,y,level,start,stop,up):
 for i in range(max(1,start),stop):
  if (y[i-1]<=level<y[i] if up else y[i-1]>=level>y[i]):
   fraction=(level-y[i-1])/(y[i]-y[i-1]);return float(t[i-1]+fraction*(t[i]-t[i-1])),i
 raise ValueError('No crossing')
for size,folder in [(20,'20mm_red'),(30,'30mm_blue'),(40,'40mm_black')]:
 for split in ['Train','Test']:
  for p in sorted((B/folder/split).glob('grasp_*/camera.csv')):
   with p.open(encoding='utf-8-sig',newline='') as f:r=list(csv.DictReader(f))
   try:
    t=np.array([float(x['t_camera']) for x in r]);t-=t[0];y=np.array([float(x['finger_width_mm']) for x in r]);assert np.isfinite([*t,*y]).all() and (np.diff(t)>0).all()
    peak=int(np.argmax(y));initial=float(np.median(y[:10]));hold=float(np.median(y[-10:]));top=float(y[peak]);assert top>max(initial,hold)
    rise_level=initial+.1*(top-initial);end_level=hold+.1*(top-hold)
    rise,ri=crossing(t,y,rise_level,1,peak+1,True);end,ei=crossing(t,y,end_level,peak+1,len(t),False)
    assert rise<t[peak]<end<=t[-1]
    for event,sec,aperture,idx in [('rise_start',rise,rise_level,ri),('peak',float(t[peak]),top,peak),('closing_complete',end,end_level,ei)]:
     rows.append(dict(object_mm=size,split=split,trial=p.parent.name,event=event,time_s=sec,normalized_time=sec/float(t[-1]),aperture_mm=aperture,duration_s=float(t[-1]),source_frame=int(r[idx]['frame']),source_csv=str(p),interpolated=event!='peak'))
   except (ValueError,AssertionError,KeyError) as e:issues.append(dict(source=str(p),issue=str(e)))
summary=[]
for size in [20,30,40]:
 for event in ['rise_start','peak','closing_complete']:
  selected=[r for r in rows if r['object_mm']==size and r['event']==event];s=dict(object_mm=size,event=event,n=len(selected))
  for field in ['time_s','normalized_time','aperture_mm']:
   a=np.array([r[field] for r in selected]);s.update({field+'_mean':float(np.mean(a)),field+'_sd':float(np.std(a,ddof=1)),field+'_median':float(np.median(a)),field+'_q25':float(np.percentile(a,25)),field+'_q75':float(np.percentile(a,75))})
  summary.append(s)
for name,data in [('event_times_per_trial.csv',rows),('event_times_summary.csv',summary)]:
 with (O/name).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
result=dict(definition=dict(rise_start='First upward crossing of initial + 10%*(peak-initial) before global peak.',peak='First frame at global maximum recorded aperture.',closing_complete='First downward crossing of hold + 10%*(peak-hold) after peak; threshold-based phase end, not physical contact detection.',initial='Median of first 10 frames.',hold='Median of last 10 frames.',time='t_camera minus first timestamp; normalized_time=time_s/trial duration.',interpolation='Linear interpolation only to locate threshold crossings. Peak is a recorded frame. Source frame for threshold events is the right bracketing frame.',sources='Current Train and Test, 100 trials each object. Includes previously manually edited 20mm Train/grasp_051.'),summary=summary,issues=issues)
(O/'event_times_method_and_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
