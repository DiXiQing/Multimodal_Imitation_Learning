from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfilt,sosfilt_zi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
base=root/'Data/临时数据/D13/3/TRIAL_Size_3cm_20261013_154107_842077_segment09'
out=base/'filter_comparison'
out.mkdir(exist_ok=True)
files=[base/n for n in ['camera.csv','imu.csv','grasp_data.csv','grasp_data_smoothed.csv']]
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
imu=pd.read_csv(base/'imu.csv').sort_values('t_imu').drop_duplicates('t_imu',keep='last')
camera=pd.read_csv(base/'camera.csv')
current=pd.read_csv(base/'grasp_data_smoothed.csv')
raw=pd.read_csv(base/'grasp_data.csv')
ti=imu.t_imu.to_numpy();tc=camera.t_camera.to_numpy()
axes=imu[['ax','ay','az']].to_numpy()
fs=50.0
# BLE receive timestamps are irregular: create a uniform 50 Hz timeline first.
uniform=ti[0]+np.arange(int(np.floor((ti[-1]-ti[0])*fs))+1)/fs
u=np.stack([np.interp(uniform,ti,axes[:,j]) for j in range(3)],axis=1)
def camera_axes(filtered):
    return np.stack([np.interp(tc,uniform,filtered[:,j]) for j in range(3)],axis=1)
variants={'Current EMA after camera sync':current[['ax','ay','az']].to_numpy()}
# A fair placement-only control: equal EMA time constant, not equal span.
camera_rate=(len(tc)-1)/(tc[-1]-tc[0])
alpha=1-(1-1/3)**(camera_rate/fs)
matched=np.empty_like(u);matched[0]=u[0]
for i in range(1,len(u)):matched[i]=alpha*u[i]+(1-alpha)*matched[i-1]
variants['Time-matched EMA before sync']=camera_axes(matched)
for cutoff in [3,5,8]:
    sos=butter(2,cutoff,fs=fs,output='sos')
    filtered=np.column_stack([sosfilt(sos,u[:,j],zi=sosfilt_zi(sos)*u[0,j])[0] for j in range(3)])
    variants[f'Causal low-pass {cutoff} Hz']=camera_axes(filtered)
    d=current.copy()
    d[['ax','ay','az']]=variants[f'Causal low-pass {cutoff} Hz']
    d['mag']=np.linalg.norm(d[['ax','ay','az']],axis=1)
    # Unrelated columns retained for comparison; velocity is NOT recomputed.
    d.to_csv(out/f'comparison_lowpass_{cutoff}Hz.csv',index=False)
t=tc-tc[0]
colors=['tab:orange','tab:purple','tab:green','tab:blue','tab:red']
fig,axs=plt.subplots(3,1,figsize=(14,11),sharex=True)
axs[0].plot(ti-tc[0],np.linalg.norm(axes,axis=1),color='gray',alpha=.55,lw=1,label='Recorded IMU magnitude (before software smoothing)')
for (label,a),color in zip(variants.items(),colors):
    mag=np.linalg.norm(a,axis=1)
    if label=='Time-matched EMA before sync':continue
    axs[0].plot(t,mag,label=label,color=color,lw=1.8)
axs[0].set_ylabel('Acceleration magnitude (m/s2)')
axs[0].legend(fontsize=9,ncol=2)
axs[0].set_title('Segment 09: filters applied to gravity-removed axes; causal filters only')
for j,label in enumerate(['ax','ay','az']):
    axs[1].plot(t,current[label],color=plt.cm.tab10(j),ls='--',alpha=.65,label=f'{label}: current')
    axs[1].plot(t,variants['Causal low-pass 5 Hz'][:,j],color=plt.cm.tab10(j),label=f'{label}: 5 Hz low-pass')
axs[1].set_ylabel('Signed axis acceleration (m/s2)');axs[1].legend(ncol=3,fontsize=9)
axs[2].plot(t,np.linalg.norm(variants['Current EMA after camera sync'],axis=1),color='tab:orange',label='Current EMA after sync')
axs[2].plot(t,np.linalg.norm(variants['Time-matched EMA before sync'],axis=1),color='tab:purple',label='EMA before sync (same time constant)')
axs[2].set_ylabel('Acceleration magnitude (m/s2)');axs[2].legend()
axs[2].set_xlabel('Time from crop start (s)')
for ax in axs:ax.grid(alpha=.2);ax.set_xlim(t[0],t[-1])
fig.tight_layout();fig.savefig(out/'segment09_filter_comparison.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(13,5))
ax.plot(t,np.linalg.norm(axes if False else raw[['ax','ay','az']].to_numpy(),axis=1),color='gray',alpha=.5,lw=1,label='Unsmooth camera-synced magnitude')
for label in ['Current EMA after camera sync','Causal low-pass 3 Hz','Causal low-pass 5 Hz']:
    a=variants[label];ax.plot(t,np.linalg.norm(a,axis=1),lw=2,label=label)
ax.set(xlabel='Time from crop start (s)',ylabel='Acceleration magnitude (m/s2)',title='Segment 09: current smoothing vs pre-sync low-pass')
ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(out/'segment09_filter_simple.png',dpi=160);plt.close(fig)
rows=[]
for label,a in variants.items():
    mag=np.linalg.norm(a,axis=1)
    rows.append(dict(method=label,peak=float(mag.max()),peak_time_s=float(t[mag.argmax()]),
                     mean_abs_second_difference=float(np.mean(np.abs(np.diff(mag,n=2)))),
                     initial_mean=float(mag[:10].mean()),final_mean=float(mag[-10:].mean())))
pd.DataFrame(rows).to_csv(out/'filter_metrics.csv',index=False)
(out/'method.json').write_text(json.dumps(dict(source=base.name,uniform_imu_rate_hz=fs,
    measured_camera_rate_hz=camera_rate,imu_median_interval_seconds=float(np.median(np.diff(ti))),
    imu_max_gap_seconds=float(np.diff(ti).max()),filter='2nd-order causal Butterworth SOS; initialized at first sample; no future samples; no post-sync EMA',
    placement_control_alpha=alpha,comparison_csv_warning='Only acceleration columns changed; velocity retained and not suitable as recomputed velocity',source_sha256=hashes),indent=2),encoding='utf-8')
assert hashes=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
print(pd.DataFrame(rows).to_string(index=False));print('camera rate',camera_rate,'IMU dt median',np.median(np.diff(ti)),'max gap',np.diff(ti).max())
print(out)
