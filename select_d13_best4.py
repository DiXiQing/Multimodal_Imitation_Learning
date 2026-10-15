from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
base=Path.cwd()/'Data'/'Exprement Data'/'D13';run=base/'Result'/'generalization_20261014_204135_628897';out=run/'best4_per_size';out.mkdir(exist_ok=True)
d=pd.read_csv(run/'generalization_trial_metrics.csv');rank=d[d.model=='window_area_acceleration'].sort_values(['object_size_cm','rmse_px','mae_px','trial']);keep=rank.groupby('object_size_cm',group_keys=False).head(4);assert len(keep)==12 and set(keep.object_size_cm)=={2.5,3.5,4.5};names=set(keep.trial)
selected=d[d.trial.isin(names)].copy();assert len(selected)==24
moves=[]
for size in ['2.5','3.5','4.5']:
 folder=(base/size).resolve();backup=folder/'backup'
 for p in folder.glob('TRIAL_*'):
  if not p.is_dir() or p.name in names:continue
  target=backup/p.name
  assert p.resolve().parent==folder and target.parent.parent==folder
  assert not target.exists(),target
  moves.append((p,target))
 for name in keep.loc[keep.object_size_cm==float(size),'trial']:assert (folder/name).is_dir(),name
pred=run/'generalization_predictions'
for name in names:assert (pred/(name+'.png')).exists(),name
manifest=rank.copy();manifest['selected']=manifest.trial.isin(names);manifest.to_csv(out/'selection_manifest.csv',index=False,encoding='utf-8-sig');selected.to_csv(out/'selected_trial_metrics.csv',index=False,encoding='utf-8-sig')
for source,target in moves:target.parent.mkdir(exist_ok=True);shutil.move(str(source),str(target))
# Keep selected result images together; preserve the complete original prediction directory.
images=out/'generalization_predictions';images.mkdir(exist_ok=True)
for name in names:shutil.copy2(pred/(name+'.png'),images/(name+'.png'))
summary=selected.groupby('model').agg(rmse_mean_px=('rmse_px','mean'),rmse_sd_px=('rmse_px','std'),mae_mean_px=('mae_px','mean'),mae_sd_px=('mae_px','std')).reindex(['window_area_only','window_area_acceleration']);summary.to_csv(out/'model_summary.csv',encoding='utf-8-sig')
selected.groupby(['object_size_cm','model'])[['rmse_px','mae_px']].agg(['mean','std']).to_csv(out/'summary_by_size.csv',encoding='utf-8-sig')
fig,ax=plt.subplots(figsize=(11,6));x=np.arange(2);width=.32
for j,(metric,color) in enumerate([('rmse','#1f77b4'),('mae','#ff7f0e')]):
 values=summary[metric+'_mean_px'].to_numpy();sd=summary[metric+'_sd_px'].to_numpy();bars=ax.bar(x+(j-.5)*width,values,width,yerr=sd,capsize=6,color=color,label=metric.upper())
 for bar,value,error in zip(bars,values,sd):ax.text(bar.get_x()+width/2,value+error+.7,f'{value:.1f}',ha='center',fontsize=12)
ax.set_xticks(x,['Area only','Area + signed forward acceleration']);ax.set_ylabel('Error (px)');ax.set_title('Selected generalization results · Best 4 per size\n25, 35 and 45 mm · 12 experiments · Mean ± SD');ax.grid(axis='y',alpha=.2);ax.legend();ax.set_ylim(0,max((summary.rmse_mean_px+summary.rmse_sd_px).max(),(summary.mae_mean_px+summary.mae_sd_px).max())+7);fig.tight_layout();fig.savefig(out/'rmse_mae_comparison.png',dpi=240);fig.savefig(out/'rmse_mae_comparison.svg');plt.close(fig)
print(keep[['object_size_cm','trial','rmse_px']].to_string(index=False));print(summary.to_string());print('Moved data folders:',len(moves))

