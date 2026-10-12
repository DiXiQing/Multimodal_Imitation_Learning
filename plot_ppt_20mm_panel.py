from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from train_window_mlp import predict_trial
root=Path.cwd();run=root/'generalization_20261012_111640_003207';out=run/'ppt_panels';out.mkdir(exist_ok=True)
metrics=pd.read_csv(run/'test_trial_metrics.csv');part=metrics[metrics.object_size_cm==2];names=sorted(part.trial.unique());assert len(names)==5
plt.rcParams.update({'font.size':12,'axes.titlesize':14})
fig,axes=plt.subplots(2,3,figsize=(16,8.2));colors=['#1f77b4','#ff7f0e','#2ca02c'];models=['window_area_only','window_area_acceleration']
for i,(ax,name) in enumerate(zip(axes.flat,names)):
 d=pd.read_csv(root/'Data'/'临时数据'/'D10'/'2'/name/'grasp_data_smoothed.csv');d['acceleration_magnitude']=np.linalg.norm(d[['ax','ay','az']],axis=1)
 x=100*(d.frame-d.frame.iloc[0])/(d.frame.iloc[-1]-d.frame.iloc[0]);ax.plot(x,d.finger_width,color=colors[0],lw=2.1)
 errors=[]
 for j,model in enumerate(models):
  target,pred,frames=predict_trial(run/'models'/f'{model}.pth',d,torch.device('cpu'));rmse=np.sqrt(np.mean((target-pred)**2));saved=part[(part.trial==name)&(part.model==model)].rmse_px.iloc[0];assert abs(rmse-saved)<.01,(name,rmse,saved)
  ax.plot(100*(frames-d.frame.iloc[0])/(d.frame.iloc[-1]-d.frame.iloc[0]),pred,color=colors[j+1],lw=1.8);errors.append(saved)
 ax.set_title(f'Experiment {i+1} · {name[-6:]}',loc='left');ax.text(.03,.95,f'RMSE (px)\nArea: {errors[0]:.1f}   Area + acc.: {errors[1]:.1f}',transform=ax.transAxes,va='top',fontsize=10,bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
 ax.set(xlim=(0,100),ylim=(0,250),xticks=[0,25,50,75,100]);ax.grid(alpha=.18)
 if i//3==1:ax.set_xlabel('Recorded experiment progress (%)')
 if i%3==0:ax.set_ylabel('Fingertip aperture (px)')
summary=axes[1,2];summary.axis('off')
from matplotlib.lines import Line2D
summary.legend([Line2D([0],[0],color=c,lw=2) for c in colors],['Reference aperture','Area only','Area + acceleration'],loc='upper left',frameon=False,fontsize=13)
a=part[part.model==models[0]].rmse_px;b=part[part.model==models[1]].rmse_px
summary.text(.02,.56,'RMSE across all 5 experiments',fontsize=14,weight='bold',transform=summary.transAxes)
summary.text(.02,.41,f'Area only:  {a.mean():.1f} ± {a.std():.1f} px\n\nArea + acceleration:  {b.mean():.1f} ± {b.std():.1f} px',fontsize=14,transform=summary.transAxes)
summary.text(.02,.12,f'Mean RMSE reduction: {(1-b.mean()/a.mean())*100:.1f}%',fontsize=14,color=colors[2],transform=summary.transAxes)
fig.suptitle('Standard test · 20 mm · All five held-out experiments',fontsize=19,weight='bold',y=.99);fig.tight_layout(rect=(0,0,1,.95),h_pad=2,w_pad=1.5)
fig.savefig(out/'standard_20mm_five_experiments.png',dpi=240);fig.savefig(out/'standard_20mm_five_experiments.svg');plt.close(fig)
print(out/'standard_20mm_five_experiments.png')
