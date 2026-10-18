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
OUT=ROOT/'Report/diagnostic_105217'
OUT.mkdir(parents=True,exist_ok=True)
NAME='TRIAL_Size_3cm_20261016_105217'
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
_,area_prediction,_=predict_trial(RUN/'models/window_area_only.pth',target,torch.device('cpu'))
px=100*(pframes-target.frame.iloc[0])/(target.frame.iloc[-1]-target.frame.iloc[0])
late=np.flatnonzero(px>=72)
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
        ax.plot(x,target[col],color='#dc2626',lw=2.6,zorder=6,label='Selected trial: 105217')
        ax.set_ylabel(label)
        if col=='forward_acceleration':ax.axhline(0,color='#64748b',alpha=.3,lw=.7)
for column in range(2):
    ax=axs[3,column]
    ax.plot(x,target.finger_width,color='#2563eb',lw=2,label='Measured FW')
    ax.plot(px,prediction,color='#16a34a',lw=2,label='OA + acceleration prediction')
    ax.plot(px,area_prediction,color='#f97316',lw=1.6,alpha=.8,label='OA-only prediction')
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

fig,axs=plt.subplots(4,2,figsize=(15,12),sharex=True,sharey='row')
summary=[]
for column,suffix in enumerate(['105159','105217']):
    name=f'TRIAL_Size_3cm_20261016_{suffix}'
    _,d,xx=next(t for t in trials if t[0]==name)
    _,pred,frames=predict_trial(checkpoint,d,torch.device('cpu'))
    _,area,_=predict_trial(RUN/'models/window_area_only.pth',d,torch.device('cpu'))
    progress=100*(frames-d.frame.iloc[0])/(d.frame.iloc[-1]-d.frame.iloc[0])
    eligible=np.flatnonzero(progress>=(75 if suffix=='105159' else 72))
    k=eligible[np.argmax(pred[eligible])]
    idx=int(np.flatnonzero(d.frame.to_numpy()==frames[k])[0])
    left=xx[idx-window+1];right=xx[idx]
    for row,(col,label) in enumerate(zip(cols,labels)):
        ax=axs[row,column]
        for other,dd,xxx in trials:
            if other!=name:ax.plot(xxx,dd[col],color='#94a3b8',alpha=.12,lw=.7)
        ax.plot(grid,means[col],color='#475569',alpha=.45,ls='--',lw=1.5,label=f'Mean (n={len(trials)})')
        ax.plot(xx,d[col],color='#dc2626',lw=2.6,label=f'Highlighted: {suffix}')
        ax.set_ylabel(label)
        if row==2:ax.axhline(0,color='#64748b',lw=.7,alpha=.4)
    ax=axs[3,column]
    ax.plot(xx,d.finger_width,color='#2563eb',lw=2,label='Measured FW')
    ax.plot(progress,area,color='#f97316',lw=1.7,label='OA only')
    ax.plot(progress,pred,color='#16a34a',lw=2,label='OA + acceleration')
    ax.scatter([right],[pred[k]],s=35,color='#16a34a',zorder=8)
    ax.set_ylabel('FW prediction (px)');ax.set_xlabel('Recorded trial progress (%)');ax.legend(fontsize=8,loc='upper left')
    axs[0,column].set_title(f'{suffix} | false peak at {right:.1f}%')
    axs[0,column].legend(fontsize=8,loc='upper left')
    for ax in axs[:,column]:
        ax.axvspan(left,right,color='#eab308',alpha=.10,zorder=0)
        ax.axvline(right,color='#a16207',lw=1.1,ls=':');ax.grid(alpha=.15);ax.set_xlim(0,100)
    trough=int(np.argmin(d.forward_acceleration))
    summary.append(dict(trial=name,second_peak_frame=int(frames[k]),second_peak_progress=right,predicted_fw=float(pred[k]),measured_fw=float(d.finger_width.iloc[idx]),area_only_fw=float(area[k]),input_first_frame=int(d.frame.iloc[idx-window+1]),input_last_frame=int(frames[k]),acc_trough=float(d.forward_acceleration.iloc[trough]),acc_trough_progress=float(xx[trough]),acc_at_second_peak=float(d.forward_acceleration.iloc[idx]),oa_at_second_peak=float(d.object_area.iloc[idx])))
fig.suptitle('Highlighted inputs and measured FW | two test trials\nYellow: 15-frame input window at the marked secondary prediction peak; no input changes',fontsize=12)
fig.tight_layout(rect=(0,0,1,.94));fig.savefig(OUT/'compare_105159_105217.png',dpi=160);plt.close(fig)
pd.DataFrame(summary).to_csv(OUT/'two_trial_summary.csv',index=False)
print(pd.DataFrame(summary).to_string(index=False))

