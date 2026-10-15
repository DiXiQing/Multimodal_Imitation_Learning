from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, sosfilt_zi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
out = root / 'Report/comparison_D13_D14_4cm'
out.mkdir(parents=True, exist_ok=True)
sos = butter(2, 3, fs=50, output='sos')
records = []
curves = {}
for group in ['D13', 'D14']:
    curves[group] = []
    for trial in sorted((root / 'Data/Exprement Data' / group / '4').glob('TRIAL*')):
        d = pd.read_csv(trial / 'grasp_data_smoothed.csv')
        imu = pd.read_csv(trial / 'imu.csv').sort_values('t_imu', kind='stable').drop_duplicates('t_imu', keep='last')
        tc = d.t_camera.to_numpy()
        ti = imu.t_imu.to_numpy()
        grid = ti[0] + np.arange(int(np.ceil((tc[-1]-ti[0])*50))+1)/50
        u = np.interp(grid, ti, imu.forward_raw)
        filtered = sosfilt(sos, u, zi=sosfilt_zi(sos)*u[0])[0]
        expected = np.interp(tc, grid, filtered)
        a = d.forward_acceleration.to_numpy()
        t = tc-tc[0]
        peak = int(np.argmax(d.finger_width))
        i, j = int(np.argmax(a)), int(np.argmin(a))
        records.append(dict(group=group, trial=trial.name, frames=len(d), duration_s=t[-1],
            positive_peak=a[i], negative_peak=a[j], peak_to_peak=a[i]-a[j], rms=np.sqrt(np.mean(a*a)),
            initial_300ms_mean=np.mean(a[t<=.3]), positive_peak_time_s=t[i], negative_peak_time_s=t[j],
            positive_peak_progress=t[i]/t[-1]*100, negative_peak_progress=t[j]/t[-1]*100,
            negative_peak_after_fwpeak_s=t[j]-t[peak], smoothing_max_error=np.max(np.abs(expected-a)),
            imu_max_gap_s=np.diff(ti).max(), forward_x=imu.forward_x.median(), forward_y=imu.forward_y.median(),forward_z=imu.forward_z.median()))
        curves[group].append((t,a,t[peak]))
metrics = pd.DataFrame(records)
metrics.to_csv(out/'per_trial_metrics.csv',index=False)
cols=['duration_s','positive_peak','negative_peak','peak_to_peak','rms','initial_300ms_mean','positive_peak_progress','negative_peak_progress','negative_peak_after_fwpeak_s']
summary=metrics.groupby('group')[cols].agg(['mean','std','median','min','max'])
summary.to_csv(out/'summary.csv')
print('counts',metrics.groupby('group').size().to_dict())
print(summary.to_string())
print('filter error',metrics.groupby('group').smoothing_max_error.max().to_dict())
print('direction',metrics.groupby('group')[['forward_x','forward_y','forward_z']].mean().to_string())
print('peak relative variability',metrics.groupby('group').positive_peak.std()/metrics.groupby('group').positive_peak.mean())
fig,axs=plt.subplots(3,1,figsize=(12,11))
for group,color in [('D13','#2563eb'),('D14','#e87918')]:
    norm=np.linspace(0,100,201)
    arr=[]
    relative=np.linspace(-1.3, .7, 201)
    aligned=[]
    for t,a,peak in curves[group]:
        x=t/t[-1]*100
        axs[0].plot(x,a,color=color,alpha=.2,lw=.8)
        arr.append(np.interp(norm,x,a))
        v=np.interp(relative,t-peak,a,left=np.nan,right=np.nan)
        aligned.append(v)
    arr=np.array(arr)
    axs[0].plot(norm,arr.mean(axis=0),color=color,lw=2.5,label=f'{group} (n={len(arr)})')
    axs[1].plot(norm,arr.mean(axis=0),color=color,lw=2,label=group)
    axs[1].fill_between(norm,arr.mean(axis=0)-arr.std(axis=0,ddof=1),arr.mean(axis=0)+arr.std(axis=0,ddof=1),color=color,alpha=.16)
    aligned=np.array(aligned)
    count=np.sum(np.isfinite(aligned),axis=0)
    good=count>=int(np.ceil(.8*len(aligned)))
    axs[2].plot(relative[good],np.nanmean(aligned[:,good],axis=0),color=color,lw=2,label=group)
    mean=np.nanmean(aligned[:,good],axis=0);std=np.nanstd(aligned[:,good],axis=0,ddof=1)
    axs[2].fill_between(relative[good],mean-std,mean+std,color=color,alpha=.16)
for ax in axs:
    ax.axhline(0,color='black',lw=.7);ax.grid(alpha=.2);ax.legend();ax.set_ylabel('Signed forward acceleration (m/s²)')
axs[0].set(title='All trials and group means',xlabel='Segment elapsed time (%)')
axs[1].set(title='Group mean ± 1 SD (time normalized; no amplitude adjustment)',xlabel='Segment elapsed time (%)')
axs[2].axvline(0,color='black',ls='--',lw=.7)
axs[2].set(title='Aligned at maximum fingertip aperture; mean ± 1 SD',xlabel='Time from maximum FW (s)')
fig.tight_layout();fig.savefig(out/'signed_acceleration_comparison.png',dpi=150);plt.close(fig)
