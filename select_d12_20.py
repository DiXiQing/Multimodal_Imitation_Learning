from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
base=Path.cwd()/'Data'/'临时数据'/'D12'/'3';paths=sorted(p for p in base.glob('TRIAL_*') if p.is_dir());grid=np.linspace(0,100,201);curves=[]
for p in paths:
 d=pd.read_csv(p/'grasp_data_smoothed.csv');x=100*(d.t_camera-d.t_camera.iloc[0])/(d.t_camera.iloc[-1]-d.t_camera.iloc[0]);curves.append([np.interp(grid,x,y) for y in [d.finger_width,d.object_area,np.linalg.norm(d[['ax','ay','az']],axis=1)]])
a=np.asarray(curves);assert len(a)>=20
scale=np.percentile(a,95,axis=(0,2))-np.percentile(a,5,axis=(0,2));z=a/scale[None,:,None];dist=np.mean((z[:,None]-z[None,:])**2,axis=(2,3))
best=None;bestscore=np.inf
for center in range(len(paths)):
 selected=list(np.argsort(dist[center])[:20]);score=dist[np.ix_(selected,selected)].sum()
 while True:
  candidate=None;newscore=score
  for old in selected:
   rest=[k for k in selected if k!=old]
   for new in range(len(paths)):
    if new in selected:continue
    value=score-2*dist[old,rest].sum()+2*dist[new,rest].sum()
    if value<newscore-1e-12:newscore=value;candidate=(old,new)
  if candidate is None:break
  selected.remove(candidate[0]);selected.append(candidate[1]);score=newscore
 if score<bestscore:bestscore=score;best=selected
keep=set(best);reserve=(base/'备用').resolve();reserve.mkdir(exist_ok=True);assert reserve.parent==base.resolve()
manifest=pd.DataFrame([dict(experiment=p.name,retained=i in keep,mean_distance_to_selected=float(dist[i,best].mean())) for i,p in enumerate(paths)])
manifest.to_csv(base/'selection_20_manifest.csv',index=False,encoding='utf-8-sig')
for i,p in enumerate(paths):
 if i not in keep:
  assert p.resolve().parent==base.resolve();target=reserve/p.name;assert not target.exists();shutil.move(str(p),str(target))
assert len(list(base.glob('TRIAL_*/grasp_data_smoothed.csv')))==20
fig,axes=plt.subplots(1,3,figsize=(18,4.5))
for j,i in enumerate(sorted(best)):
 for ax,y in zip(axes,a[i]):ax.plot(grid,y,color=plt.cm.tab20(j/19),lw=1.1)
for ax,y,label in zip(axes,a[best].mean(axis=0),['Fingertip aperture (px)','Object area (px²)','Acceleration magnitude (m/s²)']):
 ax.plot(grid,y,'k--',lw=2,label='Mean');ax.set(xlim=(0,100),xlabel='Recorded experiment progress (%)',ylabel=label);ax.grid(alpha=.2);ax.legend()
fig.tight_layout();fig.savefig(base/'three_signals_overlay.png',dpi=240);plt.close(fig)
print('Original:',len(paths),'Retained:',len(best),'Moved:',len(paths)-len(best));print(manifest[~manifest.retained].experiment.to_string(index=False))
