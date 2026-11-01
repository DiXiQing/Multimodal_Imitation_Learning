"""Export representative measured aperture-time curves for three cube sizes."""
import os,csv,json
from pathlib import Path
import numpy as np
os.environ['MPLCONFIGDIR']=str(Path(__file__).parent/'.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
base=Path(__file__).resolve().parents[1]/'Data/1029_D1'
out=base/'report_aperture_examples';out.mkdir(exist_ok=True)
selected=[]
for size,folder in [(20,'20mm_red'),(30,'30mm_blue'),(40,'40mm_black')]:
 candidates=[]
 for trial in sorted((base/folder/'Test').glob('grasp_*')):
  file=trial/'camera.csv'
  if not file.is_file():continue
  with file.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
  try:
   t=np.array([float(r['t_camera']) for r in rows]);t-=t[0]
   y=np.array([float(r['finger_width_mm']) for r in rows])
   scale=np.array([float(r['mm_per_px']) for r in rows])
  except (ValueError,KeyError):continue
  if not (np.isfinite(t).all() and np.isfinite(y).all() and np.isfinite(scale).all() and (scale>0).all() and (np.diff(t)>0).all()):continue
  peak=int(np.argmax(y))
  if peak<5 or peak>len(y)-6:continue
  feature=[float(t[-1]),float(y[peak]),float(np.median(y[:10])),float(np.median(y[-10:])),float(t[peak]/t[-1])]
  candidates.append(dict(trial=trial.name,file=str(file),t=t,y=y,feature=feature))
 assert candidates,folder
 features=np.array([r['feature'] for r in candidates]);median=np.median(features,axis=0);spread=np.percentile(features,75,axis=0)-np.percentile(features,25,axis=0);spread[spread<1e-8]=1
 scores=np.sum(np.abs((features-median)/spread),axis=1)
 best=candidates[int(np.argmin(scores))];best.update(size=size,split='Test',candidate_count=len(candidates),selection_score=float(np.min(scores)));selected.append(best)
fig,ax=plt.subplots(figsize=(7.2,4.5));fig.subplots_adjust(left=.12,right=.98,top=.97,bottom=.24)
for r,color in zip(selected,['#4477AA','#E69536','#16838D']):
 ax.plot(r['t'],r['y'],color=color,lw=2,label=f"{r['size']} mm")
ax.set_xlabel('Time (s)',fontsize=12);ax.set_ylabel('Finger aperture (mm)',fontsize=12);ax.tick_params(labelsize=10);ax.set_ylim(bottom=0);ax.set_xlim(left=0);ax.grid(axis='y',color='#DCE3EB',lw=.7);ax.spines[['top','right']].set_visible(False);ax.spines[['bottom','left']].set_color('#888888');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.20),ncol=3,frameon=False,fontsize=11)
for ext in ['png','svg','pdf']:fig.savefig(out/f'three_cube_aperture_time.{ext}',dpi=300,facecolor='white')
plt.close(fig)
manifest=dict(selection='One measured Test trial per cube, closest to within-object medians of duration, peak aperture, starting aperture, final aperture and relative peak time, scored using sum of absolute IQR-scaled deviations. Selection uses no model errors.',plot='Original recorded finger_width_mm vs t_camera minus first timestamp. No measurement smoothing, amplitude normalization or time normalization.',trials=[{k:v for k,v in r.items() if k not in ('t','y')} for r in selected])
(out/'selection_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest,indent=2))
