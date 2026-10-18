from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'Data/Exprement Data/Khairy/3'
RUN=BASE/'Result/causalCNN_20261016_141600_320259'
OUT=ROOT/'Report/diagnostic_105221'
OUT.mkdir(parents=True,exist_ok=True)
names=pd.read_csv(RUN/'data_split.csv').trial.unique()
trials={}
for name in names:
    path=BASE/name/'grasp_data_smoothed.csv'
    if not path.exists():
        paths=list(BASE.glob(f'*/{name}/grasp_data_smoothed.csv'))
        if len(paths)!=1:raise ValueError(name)
        path=paths[0]
    d=pd.read_csv(path)
    x=100*(d.frame.to_numpy()-d.frame.iloc[0])/(d.frame.iloc[-1]-d.frame.iloc[0])
    trials[name]=(d,x)
cols=['finger_width','object_area','forward_acceleration']
labels=['Measured FW (px)','Input OA (px²)','Input signed acceleration (m/s²)']
grid=np.linspace(0,100,501)
means={c:np.mean([np.interp(grid,x,d[c]) for d,x in trials.values()],axis=0) for c in cols}


def draw(suffixes,filename):
    fig,axs=plt.subplots(4,len(suffixes),figsize=(6*len(suffixes),12),sharex=True,sharey='row',squeeze=False)
    rows=[]
    for column,suffix in enumerate(suffixes):
        name=f'TRIAL_Size_3cm_20261016_{suffix}'
        d,x=trials[name]
        p=pd.read_csv(RUN/'test_predictions'/f'{name}.csv')
        px=100*(p.frame-d.frame.iloc[0])/(d.frame.iloc[-1]-d.frame.iloc[0])
        for row,(col,label) in enumerate(zip(cols,labels)):
            ax=axs[row,column]
            for other,(dd,xx) in trials.items():
                if other!=name:ax.plot(xx,dd[col],color='#94a3b8',alpha=.12,lw=.7)
            ax.plot(grid,means[col],color='#475569',alpha=.45,lw=1.7,ls='--',label=f'Mean (n={len(trials)})')
            ax.plot(x,d[col],color='#dc2626',lw=2.6,label=f'Highlighted: {suffix}')
            if column==0:ax.set_ylabel(label)
            if row==2:ax.axhline(0,color='#64748b',alpha=.35,lw=.7)
        ax=axs[3,column]
        ax.plot(x,d.finger_width,color='black',lw=2.3,label='Measured FW')
        for col,color,label in [('CNN | Area only','#2563eb','CNN: OA only'),('CNN | Area + signed forward acceleration','#dc2626','CNN: OA + acceleration')]:
            ax.plot(px,p[col],color=color,lw=1.8,label=label)
        ax.legend(loc='upper left',fontsize=8);ax.set_xlabel('Recorded trial progress (%)')
        if column==0:ax.set_ylabel('CNN prediction FW (px)')
        k=int(np.argmax(d.finger_width));trough=int(np.argmin(d.forward_acceleration))
        axs[0,column].set_title(suffix);axs[0,column].legend(fontsize=8,loc='upper left')
        for ax in axs[:,column]:ax.grid(alpha=.15);ax.set_xlim(0,100)
        rows.append(dict(trial=name,fw_peak_progress=x[k],acc_min=d.forward_acceleration.iloc[trough],acc_min_progress=x[trough],acc_at_fw_peak=d.forward_acceleration.iloc[k],oa_at_fw_peak=d.object_area.iloc[k],oa_end=d.object_area.iloc[-1]))
    fig.suptitle('Highlighted trial versus the same 30-trial reference set\nActual saved CNN predictions; progress aligned, original amplitudes retained',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.95));fig.savefig(OUT/filename,dpi=160);plt.close(fig)
    return rows


rows=draw(['105221'],'highlighted_inputs.png')
rows=draw(['105159','105217','105221'],'compare_three_trials.png')
pd.DataFrame(rows).to_csv(OUT/'comparison_summary.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
print(OUT/'compare_three_trials.png')
