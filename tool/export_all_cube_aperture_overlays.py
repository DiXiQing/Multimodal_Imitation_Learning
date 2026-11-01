"""Overlay all current Train/Test trials without smoothing or modifying sources."""
import os,csv,json
from pathlib import Path
import numpy as np
os.environ['MPLCONFIGDIR']=str(Path(__file__).parent/'.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1]/'Data/1029_D1';O=B/'report_aperture_examples'/'all_trials';O.mkdir(exist_ok=True)
groups=[];issues=[]
for size,folder,color in [(20,'20mm_red','#4477AA'),(30,'30mm_blue','#E69536'),(40,'40mm_black','#16838D')]:
 trials=[]
 for split in ['Train','Test']:
  for p in sorted((B/folder/split).glob('grasp_*/camera.csv')):
   try:
    with p.open(encoding='utf-8-sig',newline='') as f:r=list(csv.DictReader(f))
    t=np.array([float(x['t_camera']) for x in r]);t-=t[0];y=np.array([float(x['finger_width_mm']) for x in r]);cal=np.array([float(x['mm_per_px']) for x in r])
    if len(t)<2 or not np.isfinite(t).all() or not (np.diff(t)>0).all():raise ValueError('Invalid timestamps')
    if not np.isfinite(cal).all() or not (cal>0).all():raise ValueError('Missing/invalid millimetre calibration')
    valid=np.isfinite(y)
    if valid.sum()<2:raise ValueError('Insufficient measured aperture')
    if not valid.all():issues.append(dict(source=str(p),issue='Missing aperture shown as plot gaps',missing_frames=int((~valid).sum())))
    trials.append(dict(split=split,trial=p.parent.name,source=str(p),t=t,y=y))
   except (ValueError,KeyError) as e:issues.append(dict(source=str(p),issue=str(e),excluded=True))
 assert len({x['trial'] for x in trials})==len(trials),'Duplicate trial across Train/Test'
 groups.append(dict(size=size,color=color,trials=trials))
ymax=np.ceil(max(np.nanmax(r['y']) for g in groups for r in g['trials'])/20)*20
xmax=max(r['t'][-1] for g in groups for r in g['trials'])
def style(ax,normalized):
 ax.set_xlabel('Normalized time' if normalized else 'Time (s)',fontsize=12);ax.set_ylabel('Finger aperture (mm)',fontsize=12);ax.set_ylim(0,ymax);ax.set_xlim(0,1 if normalized else xmax);ax.grid(alpha=.18);ax.tick_params(labelsize=10)
def save(fig,name):
 for ext in ['png','pdf','svg']:fig.savefig(O/f'{name}.{ext}',dpi=300,facecolor='white')
 plt.close(fig)
for normalized in [True,False]:
 suffix='normalized_time' if normalized else 'actual_time'
 fig,axes=plt.subplots(1,3,figsize=(13.2,4.7),sharey=True);fig.subplots_adjust(left=.065,right=.985,bottom=.16,top=.90,wspace=.20)
 for ax,g in zip(axes,groups):
  for r in g['trials']:
   t=r['t']/r['t'][-1] if normalized else r['t'];ax.plot(t,r['y'],color=g['color'],lw=.7,alpha=.4)
  style(ax,normalized);ax.set_title(f"{g['size']} mm · {len(g['trials'])} trials",fontsize=13)
 save(fig,f'all_sizes_panels_{suffix}')
 for g in groups:
  fig,ax=plt.subplots(figsize=(5.7,4.6));fig.subplots_adjust(left=.15,right=.97,bottom=.16,top=.90)
  for r in g['trials']:
   t=r['t']/r['t'][-1] if normalized else r['t'];ax.plot(t,r['y'],color=g['color'],lw=.7,alpha=.4)
  style(ax,normalized);ax.set_title(f"{g['size']} mm · {len(g['trials'])} trials",fontsize=13);save(fig,f"{g['size']}mm_all_{suffix}")
 fig,ax=plt.subplots(figsize=(7.2,4.8));fig.subplots_adjust(left=.12,right=.98,bottom=.16,top=.97)
 for g in groups:
  for r in g['trials']:
   t=r['t']/r['t'][-1] if normalized else r['t'];ax.plot(t,r['y'],color=g['color'],lw=.65,alpha=.20)
  ax.plot([],[],color=g['color'],lw=2,label=f"{g['size']} mm (n={len(g['trials'])})")
 style(ax,normalized);ax.legend(fontsize=10,loc='upper right');save(fig,f'all_300_overlay_{suffix}')
manifest=dict(source_policy='Current camera.csv files under Train and Test only; backup, root duplicates and derived results excluded. Train+Test included once per trial. Current manually edited 20mm Train grasp_051 included as stored.',processing='Original finger_width_mm, no smoothing or aperture normalization. Normalized time=(t_camera-t_first)/(t_last-t_first). Missing aperture shown as gaps.',groups=[dict(size=g['size'],count=len(g['trials']),train=sum(r['split']=='Train' for r in g['trials']),test=sum(r['split']=='Test' for r in g['trials']),trials=[{k:v for k,v in r.items() if k not in ('t','y')} for r in g['trials']]) for g in groups],issues=issues)
(O/'overlay_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(dict(counts=[{'size':g['size'],'count':len(g['trials'])} for g in groups],issues=issues,output=str(O)),indent=2))
