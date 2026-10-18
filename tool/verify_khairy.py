from pathlib import Path
import sys
import re
from datetime import datetime,timezone,timedelta
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfilt,sosfilt_zi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from Smooth import median_then_ema
OUT=ROOT/'Data/Exprement Data/Khairy/grasp_segments'
summary=pd.read_csv(OUT/'crop_summary.csv')
sos=butter(2,3,fs=50,output='sos')
checks=[]
curves=[]
for row in summary.itertuples():
    trial=OUT/row.trial
    camera=pd.read_csv(trial/'camera.csv')
    raw=pd.read_csv(trial/'grasp_data.csv')
    smooth=pd.read_csv(trial/'grasp_data_smoothed.csv')
    imu=pd.read_csv(trial/'imu.csv').sort_values('t_imu',kind='stable').drop_duplicates('t_imu',keep='last')
    tc=camera.t_camera.to_numpy();ti=imu.t_imu.to_numpy()
    assert len(raw)==len(smooth)==len(camera)==row.video_frames
    assert np.array_equal(camera.frame,raw.frame) and np.array_equal(raw.frame,smooth.frame)
    assert np.allclose(raw.t_camera,tc,atol=1e-6,rtol=0) and np.allclose(smooth.t_camera,tc,atol=1e-6,rtol=0)
    assert ti[0]<=tc[0] and ti[-1]>=tc[-1] and np.diff(ti).max()<=.25
    assert np.array_equal(raw.forward_raw,smooth.forward_raw)
    assert np.isfinite(smooth.select_dtypes(include='number')).all().all()
    stamp=datetime.fromtimestamp(tc[0],timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')
    assert row.trial==f'TRIAL_Size_{row.size_cm:g}cm_{stamp}'
    assert re.fullmatch(r'TRIAL_Size_[\d.]+cm_\d{8}_\d{6}',row.trial)
    grid=ti[0]+np.arange(int(np.ceil((tc[-1]-ti[0])*50))+1)/50
    assert grid[-1]<=ti[-1]+1e-6
    errors={}
    for col in ['ax','ay','az','forward_raw']:
        u=np.interp(grid,ti,imu[col]);filtered=sosfilt(sos,u,zi=sosfilt_zi(sos)*u[0])[0]
        target='forward_acceleration' if col=='forward_raw' else col
        error=float(np.max(np.abs(smooth[target]-np.interp(tc,grid,filtered))))
        assert error<1e-4,(row.trial,col,error)
        errors[col]=error
    for col in ['finger_width','object_area']:assert np.allclose(smooth[col],median_then_ema(raw[col]))
    raw_error=float(np.max(np.abs(raw.forward_raw-np.interp(tc,ti,imu.forward_raw))))
    assert raw_error<1e-4,(row.trial,raw_error)
    initial_acc=float(np.median(np.abs(raw.forward_raw.iloc[:5])))
    checks.append(dict(trial=row.trial,size_cm=row.size_cm,rows=len(raw),coverage_before_s=tc[0]-ti[0],coverage_after_s=ti[-1]-tc[-1],imu_max_gap_s=np.diff(ti).max(),imu_gaps_over_60ms=int((np.diff(ti)>.06).sum()),raw_interpolation_error=raw_error,filter_max_error=max(errors.values()),forward_raw_unchanged=True,initial_5frames_abs_forward_median=initial_acc,initial_motion_review=initial_acc>.25))
    curves.append((row.size_cm,smooth))
audit=pd.DataFrame(checks);audit.to_csv(OUT/'alignment_audit.csv',index=False)
print('Counts',summary.groupby('size_cm').size().to_dict())
print('Total frames',summary.rows.sum(),'frames per trial',summary.rows.min(),summary.rows.max())
print('Preframes',summary.pre_frames.value_counts().to_dict())
print('IMU max gap',audit.groupby('size_cm').imu_max_gap_s.max().to_dict(),'gaps over60ms',audit.imu_gaps_over_60ms.sum())
print('Filter reproduction max error',audit.filter_max_error.max())
report=OUT/'verification_report.md'
with report.open('a',encoding='utf-8') as handle:
    handle.write('\nIndependent CSV checks passed for all '+str(len(summary))+' segments. Smoothing was recomputed from exported IMU for all three acceleration axes and signed forward acceleration, with maximum round-trip numerical error '+str(audit.filter_max_error.max())+' m/s². Names were checked against actual camera start timestamps in Asia/Tokyo. Counts by size: '+str(summary.groupby('size_cm').size().to_dict())+'. Total frames: '+str(summary.rows.sum())+'. Pre-roll distribution: '+str(summary.pre_frames.value_counts().to_dict())+'.\n')
    handle.write('\nBoundary selection uses 5-frame centered FW median for detection only, peak prominence 35 px and spacing 20 frames to retain smaller forward peaks obscured by larger return peaks. Onset: baseline (15th percentile within preceding 90 frames) plus 15% of opening excursion, at most 45 frames before the peak. Tail: 16 samples including stable+15, FW median between 12% and 70% of excursion, FW range <=max(18 px,10% of peak), OA relative range <=18%, OA gain >=1.22. Pre-roll up to 25 frames is shortened to avoid the previous return; accepted range is recorded in crop_ranges.csv. Contact sheets show start, peak, stable and end; participant-specific amplitude and timing are preserved. Peak candidates include returns and small adjustments, so excluded candidate counts are not counts of failed grasps.\n')
    handle.write('\nInitial motion review: '+str(int(audit.initial_motion_review.sum()))+' segments have median absolute forward acceleration >0.25 m/s² in their first five camera frames. Visual low aperture is not proof of IMU stillness; pre-roll can include early arm motion or previous movement residual. These are flagged in alignment_audit.csv, not zeroed or normalized away.\n')
print('Independent verification passed')
fig,axs=plt.subplots(3,3,figsize=(18,11),sharex=True,sharey='row')
for column,size in enumerate([3,3.5,4]):
    selected=[d for s,d in curves if s==size]
    for i,d in enumerate(selected):
        x=100*(d.t_camera-d.t_camera.iloc[0])/(d.t_camera.iloc[-1]-d.t_camera.iloc[0])
        color=plt.cm.turbo(i/max(1,len(selected)-1))
        for ax,col in zip(axs[:,column],['finger_width','object_area','forward_acceleration']):ax.plot(x,d[col],lw=.8,alpha=.55,color=color)
    axs[0,column].set_title(f'{size:g} cm | {len(selected)} grasps')
    axs[2,column].set_xlabel('Segment elapsed time (%)')
    axs[2,column].axhline(0,color='black',lw=.6)
for row,label in enumerate(['FW (px)','OA (px²)','Signed forward acceleration (m/s²)']):
    axs[row,0].set_ylabel(label)
    for ax in axs[row,:]:ax.grid(alpha=.2);ax.set_xlim(0,100)
fig.suptitle('Khairy | original amplitudes retained | median (3) + EMA (5); causal Butterworth 3 Hz at 50 Hz')
fig.tight_layout();fig.savefig(OUT/'overlay_by_size.png',dpi=150);plt.close(fig)
