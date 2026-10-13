from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];base=root/'Data'/'临时数据'/'D13'/'01_stable_calibration'
run=root/'singleSize_20261013_113528_471787';split=pd.read_csv(run/'data_split.csv').set_index('trial')
rows=[];curves={};grid=np.linspace(0,100,201)
for t in sorted(base.glob('TRIAL_*')):
    d=pd.read_csv(t/'grasp_data_smoothed.csv');r=pd.read_csv(t/'grasp_data.csv')
    x=100*(d.frame-d.frame.iloc[0])/(d.frame.iloc[-1]-d.frame.iloc[0]);acc=np.linalg.norm(d[['ax','ay','az']],axis=1)
    # Separate rising area midpoint and acceleration's first major peak.
    area=d.object_area.to_numpy();thr=area[0]+.5*(area[-1]-area[0]);at=int(np.flatnonzero(area>=thr)[0])
    cut=np.argmax(d.finger_width)
    first=int(np.argmax(acc[:cut+1]));last=cut+int(np.argmax(acc[cut:]))
    onset=np.flatnonzero(d.finger_width.to_numpy()>d.finger_width.iloc[:5].median()+.1*(d.finger_width.max()-d.finger_width.iloc[:5].median()))[0]
    camera=pd.read_csv(t/'camera.csv');start=int(r.frame.iloc[0]);pre=camera.loc[camera.frame.between(start-10,start-1)]
    rows.append(dict(trial=t.name,split=split.loc[t.name,'split'],fw_peak=float(d.finger_width.max()),fw_peak_progress=float(x.iloc[cut]),fw_onset_progress=float(x.iloc[onset]),area_mid_progress=float(x.iloc[at]),area_start=float(area[0]),area_end=float(area[-1]),acc_peak=float(acc.max()),acc_first_peak_progress=float(x.iloc[first]),acc_second_peak_progress=float(x.iloc[last]),velocity_mean=float(d.velocity.mean()),velocity_peak=float(d.velocity.max()),preceding_frames=len(pre),pre_fw_max=float(pre.finger_width.max())))
    curves[t.name]=np.array([np.interp(grid,x,y) for y in [d.finger_width,d.object_area,acc,d.velocity]])
stats=pd.DataFrame(rows);out=root/'Data'/'临时数据'/'D13'/'review';stats.to_csv(out/'peak_speed_statistics.csv',index=False)
for label,part in [('ALL17',stats),('TRAIN12',stats[stats.split=='train']),('OTHER16',stats[~stats.trial.str.endswith('110447')])]:
    print(label,part[['fw_peak','fw_peak_progress','area_mid_progress','acc_peak','acc_first_peak_progress','acc_second_peak_progress','velocity_mean','velocity_peak']].agg(['mean','std','min','max']).round(4).to_string())
print('TARGETS',stats.loc[stats.trial.str.endswith(('110207','110447'))].round(4).to_string(index=False))
print('PREFIX',stats[['trial','preceding_frames','pre_fw_max']].round(2).to_string(index=False))
target=next(n for n in curves if n.endswith('110207'));refs=stats.loc[stats.split=='train','trial'].tolist()
fig,axs=plt.subplots(2,2,figsize=(14,9))
for j,ax in enumerate(axs.flat):
    for n in refs:ax.plot(grid,curves[n][j],color='gray',lw=1,alpha=.25)
    ax.plot(grid,np.mean([curves[n][j] for n in refs],axis=0),'k--',lw=2,label='Training mean (12 trials)')
    ax.plot(grid,curves[target][j],color='red',lw=2,label='110207')
    ax.set(xlabel='Recorded trial progress (%)',ylabel=['Finger width (px)','Object area (px²)','Acceleration (m/s²)','Estimated speed (m/s)'][j]);ax.grid(alpha=.2);ax.legend(fontsize=9)
fig.tight_layout();fig.savefig(out/'110207_vs_training.png',dpi=150);plt.close(fig)
