from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
root=Path(__file__).resolve().parents[1];base=root/'Data'/'临时数据'/'D13'/'4'
out=base/'outlier_review';out.mkdir(exist_ok=True);grid=np.linspace(0,100,201)
trials=sorted(t for t in base.glob('TRIAL_*') if (t/'grasp_data_smoothed.csv').exists())
all_curves=[];rows=[]
for t in trials:
    d=pd.read_csv(t/'grasp_data_smoothed.csv');x=100*(d.frame-d.frame.iloc[0])/(d.frame.iloc[-1]-d.frame.iloc[0])
    a=np.linalg.norm(d[['ax','ay','az']],axis=1)
    ys=np.array([np.interp(grid,x,y) for y in [a,d.finger_width,d.object_area]])
    all_curves.append(ys)
    peaks,_=find_peaks(ys[0],prominence=.12,distance=20)
    rows.append(dict(trial=t.name,initial=float(np.mean(ys[0,:21])),final=float(np.mean(ys[0,-21:])),acc_peak=float(a.max()),peak_progress=float(x.iloc[np.argmax(a)]),acc_peaks=';'.join(f'{grid[k]:.1f}' for k in peaks),fw_peak=float(d.finger_width.max())))
a=np.array(all_curves)
# Compare with the median of the other trials, preserving actual amplitudes.
for i,row in enumerate(rows):
    ref=np.median(np.delete(a,i,axis=0),axis=0)
    row['acc_rmse']=float(np.sqrt(np.mean((a[i,0]-ref[0])**2)))
    row['early_rmse']=float(np.sqrt(np.mean((a[i,0,:81]-ref[0,:81])**2)))
    row['late_rmse']=float(np.sqrt(np.mean((a[i,0,81:]-ref[0,81:])**2)))
    row['tail_deviation']=abs(row['final']-float(ref[0,-21:].mean()))
    row['fw_rmse']=float(np.sqrt(np.mean((a[i,1]-ref[1])**2)))
    row['oa_rmse']=float(np.sqrt(np.mean((a[i,2]-ref[2])**2)))
    # Tail differences count, but acceleration over the whole action dominates.
    row['priority_score']=row['acc_rmse']+.25*row['tail_deviation']
ranking=pd.DataFrame(rows).sort_values('priority_score',ascending=False)
ranking.to_csv(out/'acceleration_ranking.csv',index=False)
chosen=ranking.head(5).trial.tolist();fig,axs=plt.subplots(5,3,figsize=(18,17),sharex=True)
for row,name in enumerate(chosen):
    i=[t.name for t in trials].index(name);ref=np.median(np.delete(a,i,axis=0),axis=0)
    for j,ax in enumerate(axs[row]):
        for k in range(len(a)):
            if k!=i:ax.plot(grid,a[k,j],color='gray',alpha=.15,lw=.8)
        ax.plot(grid,ref[j],'k--',lw=2,label=f'Other {len(a)-1} median')
        ax.plot(grid,a[i,j],color='tab:red',lw=2,label=name[-6:])
        ax.set_title(name[-6:]+' — '+['Acceleration','FW','OA'][j]);ax.set_ylabel(['Acceleration (m/s²)','Finger width (px)','Object area (px²)'][j]);ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_xlim(0,100)
        if row==4:ax.set_xlabel('Recorded trial progress (%)')
fig.tight_layout();fig.savefig(out/'five_candidates_comparison.png',dpi=135);plt.close(fig)
fig,axs=plt.subplots(2,3,figsize=(16,8),sharex=True,sharey=True)
for ax,name in zip(axs.flat,chosen):
    i=[t.name for t in trials].index(name);ref=np.median(np.delete(a,i,axis=0),axis=0)
    for k in range(len(a)):
        if k!=i:ax.plot(grid,a[k,0],color='gray',alpha=.18,lw=.8)
    ax.plot(grid,ref[0],'k--',lw=2,label='Other trials median');ax.plot(grid,a[i,0],color='red',lw=2,label=name[-6:])
    ax.set(xlabel='Trial progress (%)',ylabel='Acceleration (m/s²)',title=name[-6:],xlim=(0,100));ax.grid(alpha=.2);ax.legend(fontsize=8)
axs[1,2].axis('off');fig.tight_layout();fig.savefig(out/'five_acceleration_candidates.png',dpi=150);plt.close(fig)
print(ranking.round(4).to_string(index=False))
