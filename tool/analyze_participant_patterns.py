"""Descriptive timing comparison across three participants; never changes source trials."""
from pathlib import Path
from datetime import datetime
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'Data/Result'/f'participant_patterns_{datetime.now():%Y%m%d_%H%M%S}';out.mkdir(parents=True)
def read(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def num(r,k):
 try:return float(r[k])
 except:return np.nan
paths=[]
for split in ['Train','Validation']:
 for p in (ROOT/'Data/D1'/split).glob('*/camera.csv'):paths.append(('D',4,p))
for size in [3,4]:
 for split in ['Train','Validation','Training','Testing']:
  for p in (ROOT/f'Data/SONG1/{size}'/split).glob('*/camera.csv'):paths.append(('SONG',size,p))
for size in [3,3.5,4]:
 for p in (ROOT/f'Data/Khairy1/{size}').glob('*/camera.csv'):paths.append(('Khairy',size,p))
records=[];plots=[];issues=[]
for person,size,p in sorted(paths,key=lambda q:(q[0],q[1],str(q[2]))):
 rows=read(p);t=np.array([num(r,'t_camera') for r in rows]);t-=t[0];a=np.array([num(r,'finger_width') for r in rows]);d=np.array([num(r,'hand_object_distance') for r in rows]);area=np.array([num(r,'object_area') for r in rows])
 if len(rows)<10 or not np.isfinite(a).all() or not np.isfinite(t).all() or not np.all(np.diff(t)>0):issues.append(dict(source=str(p),reason='Invalid aperture/time'));continue
 sm=np.array([np.median(a[max(0,i-1):i+2]) for i in range(len(a))]);peak=int(np.argmax(sm));high=float(sm[peak]);base=float(np.median(sm[:min(5,peak+1)]));hold=float(np.median(sm[-5:]));amp=high-base
 if amp<10 or high-hold<5:issues.append(dict(source=str(p),reason='Insufficient opening/closing amplitude'));continue
 group='D normal' if person=='D' and p.parent.name.split('_')[-1]<'180000' else 'D slow A' if person=='D' and p.parent.name.split('_')[-1]<'193000' else 'D slow B' if person=='D' else person
 onset=next((i for i in range(peak+1) if np.all(sm[i:min(i+3,len(sm))]>=base+.1*amp)),0)
 def closure(fraction):
  level=high-fraction*(high-hold)
  return next((i for i in range(peak+1,len(sm)-2) if np.all(sm[i:i+3]<=level)),None)
 close=closure(.1);end=closure(.9)
 rec=dict(participant=person,group=group,size_cm=size,trial=p.parent.name,source=str(p),frames=len(rows),segment_duration_s=float(t[-1]),opening_onset_s=float(t[onset]),peak_time_s=float(t[peak]),peak_aperture_px=high,baseline_aperture_px=base,hold_aperture_px=hold,opening_to_peak_s=float(t[peak]-t[onset]),closing10_time_s=float(t[close]) if close is not None else None,opening_to_closing90_s=float(t[end]-t[onset]) if end is not None else None)
 for threshold in [.85,.9,.95]:
  mask=sm>=base+threshold*amp;lo=peak;hi=peak
  while lo>0 and mask[lo-1]:lo-=1
  while hi+1<len(mask) and mask[hi+1]:hi+=1
  rec[f'high_open_{int(threshold*100)}_duration_s']=float(t[hi]-t[lo])
 if end is not None:rec['high_open90_fraction_of_action']=rec['high_open_90_duration_s']/(t[end]-t[onset])
 else:rec['high_open90_fraction_of_action']=None
 for fraction in [.05,.1,.2]:
  idx=closure(fraction);key=int(fraction*100)
  rec[f'closing{key}_distance_px']=float(d[idx]) if idx is not None and np.isfinite(d[idx]) else None
  rec[f'closing{key}_remaining_distance_fraction']=float((d[idx]-d[-1])/(d[0]-d[-1])) if idx is not None and np.isfinite([d[0],d[-1],d[idx]]).all() and d[0]-d[-1]>20 else None
 rec['area_at_closing10_over_final']=float(area[close]/np.median(area[-5:])) if close is not None and np.isfinite(area[close]) and np.median(area[-5:])>0 else None
 records.append(rec);plots.append((rec,t,a,sm,d,area,onset,peak,close,end))
def write(name,rows):
 with (out/name).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write('per_trial.csv',records)
summary=[];keys=['opening_to_peak_s','high_open_90_duration_s','high_open90_fraction_of_action','opening_to_closing90_s','closing10_distance_px','closing10_remaining_distance_fraction','area_at_closing10_over_final']
for size in [3,3.5,4]:
 for group in ['D normal','D slow A','D slow B','SONG','Khairy']:
  subset=[r for r in records if r['group']==group and r['size_cm']==size]
  if not subset:continue
  for key in keys:
   v=np.array([r[key] for r in subset if r[key] is not None and np.isfinite(r[key])])
   summary.append(dict(group=group,size_cm=size,metric=key,total_trials=len(subset),eligible_trials=len(v),mean=float(np.mean(v)) if len(v) else None,sample_sd=float(np.std(v,ddof=1)) if len(v)>1 else None,median=float(np.median(v)) if len(v) else None))
write('group_summary.csv',summary)
if issues:write('excluded_unusable.csv',issues)
colors={'D normal':'#3485b8','D slow A':'#e59a32','D slow B':'#dc5555','SONG':'#2ca02c','Khairy':'#9255aa'}
fig,axes=plt.subplots(1,3,figsize=(15,4.8))
for ax,person in zip(axes,['D','SONG','Khairy']):
 seen=set()
 for rec,t,a,sm,d,area,onset,peak,close,end in plots:
  if rec['participant']!=person or rec['size_cm']!=4:continue
  group=rec['group'];ax.plot(t-t[onset],(a-rec['baseline_aperture_px'])/(rec['peak_aperture_px']-rec['baseline_aperture_px']),color=colors[group],alpha=.3,lw=1,label=group if group not in seen else None);seen.add(group)
 ax.set(title=f'{person} | 40 mm',xlabel='Time from opening onset (s)',ylabel='Within-trial scaled aperture');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(out/f'01_aperture_timing_comparison.{ext}',dpi=200)
plt.close(fig)
groups=list(colors);fig,axes=plt.subplots(1,2,figsize=(12,5))
for j,group in enumerate(groups):
 subset=[r for r in records if r['group']==group and r['size_cm']==4]
 for ax,key,label in zip(axes,['high_open_90_duration_s','high_open90_fraction_of_action'],['Near-maximum opening duration (s)','Near-maximum duration / action duration']):
  v=np.array([r[key] for r in subset if r[key] is not None]);ax.scatter(j+np.linspace(-.12,.12,len(v)),v,color=colors[group],s=15,alpha=.6);ax.errorbar(j,v.mean(),yerr=v.std(ddof=1),fmt='ko',capsize=4);ax.set_ylabel(label)
for ax in axes:ax.set_xticks(range(5),groups,rotation=20);ax.grid(axis='y',alpha=.2)
fig.tight_layout();fig.savefig(out/'02_open_duration.png',dpi=200);plt.close(fig)
# Khairy is plotted against apparent area only, never reinterpreted as distance.
fig,axes=plt.subplots(1,3,figsize=(15,4.5))
for ax,person in zip(axes,['D','SONG','Khairy']):
 for rec,t,a,sm,d,area,onset,peak,close,end in plots:
  if rec['participant']!=person or rec['size_cm']!=4:continue
  x=area if person=='Khairy' else d;ax.plot(x,a,color=colors[rec['group']],alpha=.25,lw=1)
  if close is not None:ax.scatter(x[close],a[close],color=colors[rec['group']],s=12)
 ax.set(title=person,xlabel='Object apparent area (px²)' if person=='Khairy' else 'Hand-object distance (px)',ylabel='Fingertip aperture (px)');ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(out/'03_closure_relationships.png',dpi=200);plt.close(fig)
# Six roughly representative traces per person for direct boundary review.
fig,axes=plt.subplots(3,6,figsize=(18,9))
for row,person in enumerate(['D','SONG','Khairy']):
 subset=[z for z in plots if z[0]['participant']==person and z[0]['size_cm']==4]
 for ax,i in zip(axes[row],np.linspace(0,len(subset)-1,6).astype(int)):
  rec,t,a,sm,d,area,onset,peak,close,end=subset[i];ax.plot(t,a,color='#222222');ax.axvline(t[onset],color='#3485b8',lw=.8)
  if close is not None:ax.axvline(t[close],color='#e59a32',lw=.8)
  ax.set_title(f'{person} {rec["trial"][-6:]}',fontsize=9);ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(out/'04_event_review.png',dpi=160);plt.close(fig)
(out/'method_and_limits.json').write_text(json.dumps(dict(primary='Black 40 mm, D=52 current Train+Validation, SONG=39 current Train+Validation, Khairy=55 direct trials. Also supplementary available 30 mm /35 mm results. Backups and model results excluded.',events='Centered 3-point median for event localization only. Initial aperture=first 5 median; peak=max median; final hold=last5 median. Opening onset=first 3 samples above initial+10% opening amplitude. Early closure=3 samples below peak-10%*(peak-final hold). Near-maximum duration=contiguous interval around peak above initial+90%*(peak-initial). Action endpoint=90% closure; not independently verified contact.',limitations=['Khairy has apparent area, not hand-object distance, and a different camera perspective; no pixel-scale cross-person comparison or area-to-distance substitution.','Only D has documented speed conditions; cannot establish other participants speed responses.','D/SONG sources were manually cropped/selected; groups are different sessions and not randomized matched controls.','Three-point median can suppress short peaks and timing endpoints are operational; CSV original values unchanged.','Long near-maximum aperture duration does not by itself prove stationary finger hold; slopes may remain.']),indent=2))
for r in summary:
 if r['size_cm']==4:print(r)
print('RESULT',out)
