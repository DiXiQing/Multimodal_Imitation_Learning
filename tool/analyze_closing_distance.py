"""Descriptive closing-distance event analysis, without fitting or editing trials."""
from pathlib import Path
from datetime import datetime
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'Data/D1/Train';out=ROOT/'New/Result'/f'closing_distance_analysis_{datetime.now():%Y%m%d_%H%M%S}';out.mkdir()
records=[];curves=[]
for p in sorted(source.glob('*/camera.csv')):
 rows=list(csv.DictReader(p.open(encoding='utf-8-sig')));meta=json.loads((p.parent/'segment.json').read_text(encoding='utf-8-sig'))
 t=np.array([float(r['t_camera']) for r in rows]);t-=t[0];a=np.array([float(r['finger_width']) if r['finger_width'] else np.nan for r in rows]);d=np.array([float(r['hand_object_distance']) if r['hand_object_distance'] else np.nan for r in rows]);f=np.array([int(r['frame']) for r in rows]);offset=int(meta.get('source_start_frame',f[0]))-int(meta['start']);stable=int(np.argmin(abs(f-(int(meta['stable'])+offset))));onset=int(np.argmin(abs(f-(int(meta['onset'])+offset))))
 assert np.isfinite(a).all()
 sm=np.array([np.median(a[max(0,i-1):i+2]) for i in range(len(a))]);peak=int(np.argmax(sm[:stable+1]));high=float(sm[peak]);hold=float(np.median(a[stable:]));assert high-hold>10,p
 stamp=p.parent.name.split('_')[-1];group='Fast' if stamp<'180000' else 'Slow A (19:19)' if stamp<'193000' else 'Slow B (19:32)'
 events={}
 for fraction in [.05,.10,.20]:
  level=high-fraction*(high-hold);found=None
  for i in range(peak+1,min(stable+1,len(sm)-2)):
   if np.all(sm[i:i+3]<=level):found=i;break
  if found is None:raise ValueError(f'No closure event {p} {fraction}')
  events[fraction]=found
  records.append(dict(trial=p.parent.name,group=group,closure_fraction=fraction,event_frame=int(f[found]),event_segment_frame=found+1,event_time_s=float(t[found]),closing_distance_px=float(d[found]),normalized_remaining_distance_percent=float((d[found]-d[-1])/(d[0]-d[-1])*100),peak_aperture_px=high,hold_aperture_px=hold,onset_to_stable_s=float(t[stable]-t[onset]),peak_to_closing_event_s=float(t[found]-t[peak])))
 curves.append((p.parent.name,group,t,a,d,events[.1],high-.1*(high-hold)))
summary=[]
for group in ['Fast','Slow A (19:19)','Slow B (19:32)','All slow']:
 for fraction in [.05,.1,.2]:
  part=[r for r in records if r['closure_fraction']==fraction and (r['group']==group or group=='All slow' and r['group']!='Fast')]
  for key in ['closing_distance_px','normalized_remaining_distance_percent','peak_aperture_px','onset_to_stable_s']:
   v=np.array([r[key] for r in part]);summary.append(dict(group=group,closure_fraction=fraction,metric=key,trials=len(v),mean=float(v.mean()),sample_sd=float(v.std(ddof=1)),minimum=float(v.min()),maximum=float(v.max())))
for name,data in [('per_trial_events',records),('summary',summary)]:
 with (out/(name+'.csv')).open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
fig,axes=plt.subplots(8,5,figsize=(18,20))
for ax,(name,group,t,a,d,event,level) in zip(axes.flat,curves):
 ax.plot(t,a,color='#222222',lw=1);ax.axhline(level,color='#bbbbbb',lw=.6);ax.axvline(t[event],color='#e59a32',lw=1);ax.plot(t[event],a[event],'o',color='#e59a32',ms=3);ax.set_title(f'{name[-6:]} | d={d[event]:.1f} px',fontsize=9);ax.grid(alpha=.15)
fig.tight_layout();fig.savefig(out/'closing_event_review.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4.5));groups=['Fast','Slow A (19:19)','Slow B (19:32)'];colors=['#3485b8','#e59a32','#2ca02c']
for j,(group,color) in enumerate(zip(groups,colors)):
 part=[r for r in records if r['group']==group and r['closure_fraction']==.1];v=[r['closing_distance_px'] for r in part];axes[0].scatter(j+np.linspace(-.12,.12,len(v)),v,color=color,alpha=.7);axes[0].errorbar(j,np.mean(v),yerr=np.std(v,ddof=1),fmt='ko',capsize=5)
 for name,g,t,a,d,event,level in curves:
  if g==group:axes[1].plot(d,a,color=color,alpha=.3,lw=1);axes[1].scatter(d[event],a[event],color=color,s=12)
axes[0].set_xticks(range(3),groups);axes[0].set_ylabel('Distance at 10% closure (px)');axes[1].set(xlabel='Hand-object distance (px)',ylabel='Fingertip aperture (px)')
for ax in axes:ax.grid(alpha=.2)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(out/f'closing_distance_comparison.{ext}',dpi=200)
plt.close(fig)
(out/'method.json').write_text(json.dumps(dict(source=str(source),event='Operational early-closure landmark, not exact physiological onset: first sample after smoothed peak below peak - fraction*(peak-hold) for 3 consecutive samples; peak uses centered 3-point median for event localization only. Hold is median from stored stable boundary to end. Distance is original recorded distance at landmark.',thresholds=[.05,.1,.2],primary=.1,grouping='Recording batches identified by timestamp: fast 16:17, slow A 19:19, slow B 19:32. No model training, no data edits. These are selected training data, descriptive analysis only.'),indent=2))
for r in summary:
 if r['closure_fraction']==.1:print(r)
print(out)
