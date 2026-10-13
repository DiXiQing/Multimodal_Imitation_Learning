from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
import train_window_mlp as m
result=root/'Data/Result/singleSize_20261013_155533_268862'
split=pd.read_csv(result/'data_split.csv')
trials={}
for name in split.trial:
    files=list((root/'Data/临时数据/D13/3').rglob(name+'/grasp_data_smoothed.csv'))
    assert len(files)==1,(name,files)
    d=pd.read_csv(files[0]);d['acceleration_magnitude']=np.linalg.norm(d[['ax','ay','az']],axis=1)
    trials[name]=d
grid=np.linspace(0,100,201)
cols=['finger_width','object_area','acceleration_magnitude']
def normalized(d):
    x=100*(d.frame-d.frame.iloc[0])/(d.frame.iloc[-1]-d.frame.iloc[0])
    return np.stack([np.interp(grid,x,d[c]) for c in cols])
train=np.stack([normalized(trials[n]) for n in split.loc[split.split=='train','trial']])
mean=train.mean(axis=0)
fig,axs=plt.subplots(3,3,figsize=(15,10),sharex=True)
rows=[]
for r,name in enumerate(split.loc[split.split=='test','trial']):
    d=trials[name]; curves=normalized(d)
    for j,ax in enumerate(axs[r]):
        ax.plot(grid,mean[j],color='black',label='Training mean')
        ax.fill_between(grid,train[:,j].min(axis=0),train[:,j].max(axis=0),color='gray',alpha=.15,label='Training range')
        ax.plot(grid,curves[j],color='tab:red',label=name.rsplit('_',1)[1])
        ax.set_ylabel(cols[j]);ax.grid(alpha=.2)
    axs[r,0].legend(fontsize=8)
    x=100*(d.frame-d.frame.iloc[0])/(d.frame.iloc[-1]-d.frame.iloc[0])
    for model in m.MODEL_SPECS:
        target,pred,frames=m.predict_trial(result/'models'/f'{model}.pth',d,torch.device('cpu'))
        progress=100*(frames-d.frame.iloc[0])/(d.frame.iloc[-1]-d.frame.iloc[0])
        for lo,hi in [(0,40),(40,60),(60,80),(80,101)]:
            mask=(progress>=lo)&(progress<hi)
            rows.append(dict(segment=name.rsplit('_',1)[1],model=model,stage=f'{lo}-{min(hi,100)}',rmse=float(np.sqrt(np.mean((pred[mask]-target[mask])**2))),bias=float(np.mean(pred[mask]-target[mask]))))
    print(name.rsplit('_',1)[1], 'FW peak',float(d.finger_width.max()),'at',float(x.iloc[d.finger_width.argmax()]),'acc first',float(x.iloc[np.flatnonzero(x.to_numpy()<60)[d.acceleration_magnitude[x<60].argmax()]]),'acc second',float(x.iloc[np.flatnonzero(x.to_numpy()>=60)[d.acceleration_magnitude[x>=60].argmax()]]))
for ax in axs[-1]:ax.set_xlabel('Grasp progress (%)')
fig.tight_layout();fig.savefig(result/'acceleration_input_diagnosis.png',dpi=140);plt.close(fig)
pd.DataFrame(rows).to_csv(result/'stage_error_diagnosis.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
for c in range(3):
    print('train mean',cols[c],'peak at',grid[mean[c].argmax()],'peak',mean[c].max())
