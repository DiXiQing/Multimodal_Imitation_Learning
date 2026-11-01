"""Select three real Test trials with closely matched movement timing."""
import os,csv,json,itertools
from pathlib import Path
import numpy as np
os.environ['MPLCONFIGDIR']=str(Path(__file__).parent/'.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1]/'Data/1029_D1';O=B/'report_aperture_examples';groups=[]
def cross(t,y,level,start,end,up):
 for i in range(max(1,start),end):
  if (y[i-1]<=level<y[i] if up else y[i-1]>=level>y[i]):
   return float(t[i-1]+(level-y[i-1])/(y[i]-y[i-1])*(t[i]-t[i-1]))
 raise ValueError('crossing absent')
for size,folder in [(20,'20mm_red'),(30,'30mm_blue'),(40,'40mm_black')]:
 group=[]
 for p in sorted((B/folder/'Test').glob('grasp_*/camera.csv')):
  with p.open(encoding='utf-8-sig',newline='') as f:r=list(csv.DictReader(f))
  try:
   t=np.array([float(x['t_camera']) for x in r]);t-=t[0];y=np.array([float(x['finger_width_mm']) for x in r]);d=np.array([float(x['index_mcp_object_distance_mm']) for x in r]);cal=np.array([float(x['mm_per_px']) for x in r])
   assert np.isfinite([*t,*y,*d,*cal]).all() and (cal>0).all() and (np.diff(t)>0).all()
   peak=int(np.argmax(y));low=float(np.median(y[:10]));hold=float(np.median(y[-10:]));top=float(y[peak]);d0=float(d[0]);df=float(np.median(d[-10:]));assert d0>df
   a=cross(t,y,low+.1*(top-low),1,peak+1,True);b=cross(t,y,low+.9*(top-low),1,peak+1,True)
   c=cross(t,y,hold+.9*(top-hold),peak+1,len(y),False);e=cross(t,y,hold+.1*(top-hold),peak+1,len(y),False)
   ds=cross(t,d,df+.9*(d0-df),1,len(d),False);de=cross(t,d,df+.1*(d0-df),1,len(d),False)
   features=np.array([t[-1],a,b,t[peak],c,e,ds,de])
   group.append(dict(size=size,trial=p.parent.name,source=str(p),t=t,y=y,features=features,distance_change_mm=d0-df))
  except (ValueError,AssertionError,KeyError):continue
 groups.append(group)
# Compare actual seconds, including hand-object approach timing and aperture phases.
scale=np.array([.15,.10,.10,.15,.10,.10,.10,.10])
best=None
for combo in itertools.product(*groups):
 F=np.array([x['features'] for x in combo]);score=float(np.sum(np.var(F/scale,axis=0)))
 if best is None or score<best[0]:best=(score,combo)
score,selected=best
for normalized in [False,True]:
 fig,ax=plt.subplots(figsize=(7.2,4.5));fig.subplots_adjust(left=.12,right=.98,top=.97,bottom=.24)
 for r,color in zip(selected,['#4477AA','#E69536','#16838D']):
  t=r['t']/r['t'][-1] if normalized else r['t'];ax.plot(t,r['y'],lw=2,color=color,label=f"{r['size']} mm")
 ax.set_xlabel('Normalized time' if normalized else 'Time (s)',fontsize=12);ax.set_ylabel('Finger aperture (mm)',fontsize=12);ax.tick_params(labelsize=10);ax.set_ylim(bottom=0);ax.set_xlim(left=0);ax.grid(axis='y',color='#DCE3EB',lw=.7);ax.spines[['top','right']].set_visible(False);ax.spines[['bottom','left']].set_color('#888888');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.20),ncol=3,frameon=False,fontsize=11)
 name='matched_timing_normalized_time' if normalized else 'matched_timing_actual_time'
 for ext in ['png','svg','pdf']:fig.savefig(O/f'{name}.{ext}',dpi=300,facecolor='white')
 plt.close(fig)
manifest=dict(selection='Minimum across-object variance of actual timestamps for total duration, aperture opening 10/90%, peak, closing 90/10%, distance approach 10/90% progress. All candidates from Test. No model-error selection; approximately matched timing, not identical physical velocity.',feature_labels=['total_duration_s','opening_10_s','opening_90_s','peak_s','closing_90_s','closing_10_s','approach_10_s','approach_90_s'],score=score,trials=[dict(size=r['size'],trial=r['trial'],source=r['source'],features=r['features'].tolist(),distance_change_mm=r['distance_change_mm']) for r in selected])
(O/'matched_timing_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(manifest,indent=2))
