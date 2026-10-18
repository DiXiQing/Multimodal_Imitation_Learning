from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from train_window_mlp import predict_trial

BASE=ROOT/'Data/Exprement Data/Khairy/3'
RUN=BASE/'Result/singleSize_20261016_125512_556755'
OUT=ROOT/'Report/diagnostic_105159'
OUT.mkdir(parents=True,exist_ok=True)
NAME='TRIAL_Size_3cm_20261016_105159'
split=pd.read_csv(RUN/'data_split.csv')
trials=[]
for name in split.trial.unique():
    path=BASE/name/'grasp_data_smoothed.csv'
    if not path.exists():
        matches=list(BASE.glob(f'*/{name}/grasp_data_smoothed.csv'))
        if len(matches)!=1:raise ValueError(f'Cannot uniquely locate {name}')
        path=matches[0]
    d=pd.read_csv(path)
    x=100*(d.frame.to_numpy()-d.frame.iloc[0])/(d.frame.iloc[-1]-d.frame.iloc[0])
    trials.append((name,d,x))
_,target,x=next(t for t in trials if t[0]==NAME)
checkpoint=RUN/'models/window_area_acceleration.pth'
_,prediction,pframes=predict_trial(checkpoint,target,torch.device('cpu'))
px=100*(pframes-target.frame.iloc[0])/(target.frame.iloc[-1]-target.frame.iloc[0])
late=np.flatnonzero(px>=75)
peak=late[np.argmax(prediction[late])]
frame=int(pframes[peak]);end=int(np.flatnonzero(target.frame.to_numpy()==frame)[0])
window=int(torch.load(checkpoint,map_location='cpu',weights_only=False)['window_size'])
start=end-window+1;left=float(x[start]);right=float(x[end])
grid=np.linspace(0,100,501)
cols=['finger_width','object_area','forward_acceleration']
means={col:np.mean([np.interp(grid,xx,d[col]) for _,d,xx in trials],axis=0) for col in cols}
fig,axs=plt.subplots(4,2,figsize=(15,13),sharex='col')
labels=['Measured FW (px)','Input OA (px²)','Input signed acceleration (m/s²)']
for row,(col,label) in enumerate(zip(cols,labels)):
    for column in range(2):
        ax=axs[row,column]
        for name,d,xx in trials:
            if name!=NAME:ax.plot(xx,d[col],color='#94a3b8',alpha=.15,lw=.8)
        ax.plot(grid,means[col],color='#475569',alpha=.48,lw=1.8,ls='--',label=f'All-trial mean (n={len(trials)})')
        ax.plot(x,target[col],color='#dc2626',lw=2.6,zorder=6,label='Selected trial: 105159')
        ax.set_ylabel(label)
        if col=='forward_acceleration':ax.axhline(0,color='#64748b',alpha=.3,lw=.7)
for column in range(2):
    ax=axs[3,column]
    ax.plot(x,target.finger_width,color='#2563eb',lw=2,label='Measured FW')
    ax.plot(px,prediction,color='#16a34a',lw=2,label='OA + acceleration prediction')
    ax.scatter([right],[prediction[peak]],s=35,color='#16a34a',zorder=8)
    ax.set_ylabel('FW prediction (px)')
    ax.set_xlabel('Recorded trial progress (%)')
    ax.legend(loc='upper left',fontsize=8)
for row in range(4):
    for column in range(2):
        ax=axs[row,column]
        ax.axvspan(left,right,color='#eab308',alpha=.12,zorder=0)
        ax.axvline(right,color='#a16207',ls=':',lw=1.3)
        ax.grid(alpha=.15)
        ax.set_xlim((0,100) if column==0 else (max(0,left-12),100))
axs[0,0].set_title('Full trial | selected input / target highlighted')
axs[0,1].set_title('Late-phase detail | shaded: input window for false peak')
axs[0,0].legend(loc='upper left',fontsize=8)
fig.suptitle(f'{NAME}\nYellow = {window}-frame model input at false peak; no additional filtering',fontsize=12)
fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUT/'highlighted_inputs.png',dpi=160);plt.close(fig)
pd.DataFrame({'progress_percent':grid,**means}).to_csv(OUT/'comparison_mean.csv',index=False)
target.iloc[start:end+1].to_csv(OUT/'false_peak_input_window.csv',index=False)
print('Trials:',len(trials),'false peak frame:',frame,'prediction:',prediction[peak],'measured:',target.finger_width.iloc[end],flush=True)
print('Input progress:',left,right,'frames:',int(target.frame.iloc[start]),frame,flush=True)
for col in cols:
    a=target[col].iloc[start:end+1]
    print(col,'window first / min / max / last:',a.iloc[0],a.min(),a.max(),a.iloc[-1],flush=True)
print(OUT/'highlighted_inputs.png')
