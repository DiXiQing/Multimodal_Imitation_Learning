from pathlib import Path
from datetime import datetime
import sys,json,shutil,hashlib
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfilt,sosfilt_zi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
base=root/'Data/临时数据/D13/3'
trials=sorted(p.parent for p in base.glob('TRIAL_*/grasp_data_smoothed.csv'))
assert len(trials)==15, f'Expected 15 active trials, found {len(trials)}'
backup=root/'analysis'/f'D13_3_before_lowpass3Hz_{datetime.now():%Y%m%d_%H%M%S}'
backup.mkdir()
fs=50.0
sos=butter(2,3,fs=fs,output='sos')
prepared=[]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
# Validate and calculate every trial before replacing any training input.
for trial in trials:
    old=pd.read_csv(trial/'grasp_data_smoothed.csv')
    camera=pd.read_csv(trial/'camera.csv').set_index('frame')
    imu=pd.read_csv(trial/'imu.csv').sort_values('t_imu').drop_duplicates('t_imu',keep='last')
    tc=camera.loc[old.frame,'t_camera'].to_numpy()
    ti=imu.t_imu.to_numpy()
    axes=imu[['ax','ay','az']].to_numpy()
    assert np.isfinite(axes).all() and np.all(np.diff(ti)>0)
    assert ti[0]<=tc[0] and ti[-1]>=tc[-1]
    uniform=ti[0]+np.arange(int(np.floor((ti[-1]-ti[0])*fs))+1)/fs
    u=np.column_stack([np.interp(uniform,ti,axes[:,j]) for j in range(3)])
    filtered=np.column_stack([sosfilt(sos,u[:,j],zi=sosfilt_zi(sos)*u[0,j])[0] for j in range(3)])
    updated=old.copy()
    updated[['ax','ay','az']]=np.column_stack([np.interp(tc,uniform,filtered[:,j]) for j in range(3)])
    updated['mag']=np.linalg.norm(updated[['ax','ay','az']],axis=1)
    assert np.isfinite(updated[['ax','ay','az','mag']]).all().all()
    originals={n:digest(trial/n) for n in ['camera.csv','imu.csv','grasp_data.csv']}
    prepared.append((trial,old,updated,originals,float(np.diff(ti).max())))
for trial,old,updated,originals,gap in prepared:
    saved=backup/trial.name;saved.mkdir()
    for name in ['grasp_data_smoothed.csv','smoothing_preview.png','acceleration_processing.json']:
        if (trial/name).exists():shutil.copy2(trial/name,saved/name)
    updated.to_csv(trial/'grasp_data_smoothed.csv',index=False,float_format='%.9g')
    reread=pd.read_csv(trial/'grasp_data_smoothed.csv')
    for col in old.columns:
        if col not in ['ax','ay','az','mag']:
            np.testing.assert_allclose(old[col],reread[col],rtol=1e-8,atol=1e-8)
    np.testing.assert_allclose(reread.mag,np.linalg.norm(reread[['ax','ay','az']],axis=1),rtol=1e-8)
    assert originals=={n:digest(trial/n) for n in originals}
    meta=dict(method='2nd-order causal Butterworth low-pass on gravity-removed sensor axes, before camera synchronization',
              cutoff_hz=3,uniform_imu_rate_hz=fs,initialization='steady state at first segment sample',
              irregular_timestamps='sort, deduplicate, interpolate to uniform 50 Hz',
              magnitude='norm of filtered axes after interpolation to camera times',
              post_sync_acceleration_ema=False,fw_oa='existing smoothing retained',
              velocity='existing values retained; training does not use velocity',
              max_imu_gap_seconds=gap,source_sha256=originals,backup=str(saved))
    (trial/'acceleration_processing.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    raw=pd.read_csv(trial/'grasp_data.csv')
    fig,axs=plt.subplots(3,1,figsize=(11,8),sharex=True)
    for ax,col,label in zip(axs[:2],['finger_width','object_area'],['FW (px)','OA (px2)']):
        ax.plot(raw.frame,raw[col],alpha=.4,label='Raw');ax.plot(updated.frame,updated[col],label='Smoothed')
        ax.set_ylabel(label);ax.legend()
    axs[2].plot(old.frame,np.linalg.norm(old[['ax','ay','az']],axis=1),alpha=.7,label='Previous EMA after sync')
    axs[2].plot(updated.frame,updated.mag,label='3 Hz low-pass before sync',lw=2)
    axs[2].set_ylabel('Acceleration (m/s2)');axs[2].legend();axs[2].set_xlabel('Source frame')
    for ax in axs:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(trial/'smoothing_preview.png',dpi=140);plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(16,8));signal_axes=[axs[0,0],axs[0,1],axs[1,0]]
for i,(trial,old,d,originals,gap) in enumerate(prepared):
    x=100*(d.frame-d.frame.iloc[0])/(d.frame.iloc[-1]-d.frame.iloc[0])
    for ax,col in zip(signal_axes,['finger_width','object_area','mag']):
        ax.plot(x,d[col],lw=1.2,color=plt.cm.tab20(i),label=trial.name.rsplit('_',1)[-1])
for ax,label in zip(signal_axes,['Finger width (px)','Object area (px2)','Acceleration magnitude (m/s2)']):
    ax.set(xlabel='Recorded trial progress (%)',ylabel=label,xlim=(0,100));ax.grid(alpha=.2)
axs[1,1].axis('off');h,l=signal_axes[0].get_legend_handles_labels()
axs[1,1].legend(h,l,ncol=3,loc='center',frameon=False,title='3 cm / 15 trials / acceleration low-pass 3 Hz')
fig.tight_layout();fig.savefig(base/'3cm_smooth_combined_acceleration_overlay.png',dpi=150);plt.close(fig)
print('Updated 15 trials. Original source CSVs unchanged. Backup:',backup)
sys.path.insert(0,str(root))
import train_window_mlp as m
loaded=m.load_trials(base)
assert len(loaded)==15
for name,d in loaded.items():
    for cols in m.MODEL_SPECS.values():m.make_trial_windows(d,cols,m.WINDOW_SIZE)
print('Verified: both models have valid windows in all 15 trials.')
