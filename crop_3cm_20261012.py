from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
root=Path.cwd();sys.path.insert(0,str(root))
from Analyze import calculate_3d_speed
from Smooth import smooth_trial
base=root/'Data'/'BackupData'/'D';review=base/'review_3cm_20261012';review.mkdir(exist_ok=True)
rows=[];valid=[]
for p in sorted(base.glob('TRIAL_Size_3cm_20261012_*')):
    if (p/'crop_metadata.json').exists() and (p/'grasp_data_smoothed.csv').exists():
        record=json.loads((p/'crop_metadata.json').read_text(encoding='utf-8'));rows.append(record);valid.append((p,pd.read_csv(p/'grasp_data_smoothed.csv')));continue
    record=dict(experiment=p.name,status='invalid')
    try:
        paths=[p/'camera.csv',p/'imu.csv'];hashes=[hashlib.sha256(q.read_bytes()).hexdigest() for q in paths]
        c=pd.read_csv(paths[0]);imu=pd.read_csv(paths[1]).sort_values('t_imu').drop_duplicates('t_imu',keep='last')
        if len(c)<30:raise ValueError('Too few camera samples')
        if len(c)>1000:raise ValueError('Long recording contains many opening cycles; requires separate segmentation')
        y=c.finger_width.rolling(5,center=True,min_periods=1).median().to_numpy()
        peaks,props=find_peaks(y,prominence=60,distance=20)
        if not len(peaks):raise ValueError('No complete aperture peak')
        peak=int(peaks[np.argmax(props['prominences'])]);left=int(np.argmin(y[:peak+1]));low=y[left];height=y[peak]-low
        threshold=low+.1*height
        below=np.flatnonzero(y[:peak]<threshold)
        onset=int(below[-1]+1) if len(below) else left
        start=max(0,onset-15)
        closing=np.flatnonzero(y[peak+1:]<=low+.45*height)
        if not len(closing):raise ValueError('No closing phase after peak')
        end=min(len(c)-1,peak+1+int(closing[0])+15)
        d=c.iloc[start:end+1].copy();tc=d.t_camera.to_numpy();im=imu[(imu.t_imu>=tc[0])&(imu.t_imu<=tc[-1])]
        if len(im)<10:raise ValueError('Insufficient overlapping IMU samples')
        if (im.t_imu.iloc[-1]-im.t_imu.iloc[0])/(tc[-1]-tc[0])<.9:raise ValueError('Incomplete IMU coverage')
        if np.diff(im.t_imu).max()>.25:raise ValueError('IMU gap exceeds 0.25 s')
        ti=im.t_imu.to_numpy();a=im[['ax','ay','az']].to_numpy();_,_,_,speed=calculate_3d_speed(ti,a)
        for j,col in enumerate(['ax','ay','az']):d[col]=np.interp(tc,ti,a[:,j])
        d['mag']=np.linalg.norm(d[['ax','ay','az']],axis=1);d['velocity']=np.interp(tc,ti,speed)
        if len(d)<30 or not np.isfinite(d.to_numpy()).all() or (d.object_area<=0).any():raise ValueError('Invalid cropped values or too short')
        smooth=smooth_trial(d);smooth['mag']=np.linalg.norm(smooth[['ax','ay','az']],axis=1)
        d.to_csv(p/'grasp_data.csv',index=False);smooth.to_csv(p/'grasp_data_smoothed.csv',index=False)
        fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
        for ax,col,label in zip(axes,['finger_width','object_area','mag'],['Fingertip aperture (px)','Object area (px²)','Acceleration magnitude (m/s²)']):
            ax.plot(d.frame,d[col],alpha=.4,label='Raw');ax.plot(smooth.frame,smooth[col],label='Smoothed');ax.set_ylabel(label);ax.grid(alpha=.2)
        axes[0].legend();axes[-1].set_xlabel('Frame');fig.tight_layout();fig.savefig(p/'smoothing_preview.png',dpi=150);plt.close(fig)
        record.update(status='valid',frame_start=int(d.frame.iloc[0]),frame_end=int(d.frame.iloc[-1]),samples=len(d),imu_samples=len(im),peak_aperture=float(smooth.finger_width.max()))
        (p/'crop_metadata.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        assert hashes==[hashlib.sha256(q.read_bytes()).hexdigest() for q in paths]
        valid.append((p,smooth))
    except Exception as e:record['reason']=str(e)
    rows.append(record)
summary=pd.DataFrame(rows);summary.to_csv(review/'crop_summary.csv',index=False,encoding='utf-8-sig')
fig,axes=plt.subplots(1,3,figsize=(18,4.5));grid=np.linspace(0,100,201);allcurves=[]
for i,(p,d) in enumerate(valid):
    x=100*(d.t_camera-d.t_camera.iloc[0])/(d.t_camera.iloc[-1]-d.t_camera.iloc[0]);signals=[d.finger_width,d.object_area,d.mag]
    allcurves.append([np.interp(grid,x,y) for y in signals])
    for ax,y in zip(axes,signals):ax.plot(x,y,lw=1,alpha=.75,color=plt.cm.turbo(i/max(1,len(valid)-1)))
if valid:
    mean=np.mean(allcurves,axis=0)
    for ax,y,label in zip(axes,mean,['Fingertip aperture (px)','Object area (px²)','Acceleration magnitude (m/s²)']):
        ax.plot(grid,y,'k--',lw=2,label='Mean');ax.set(xlim=(0,100),xlabel='Recorded experiment progress (%)',ylabel=label);ax.grid(alpha=.2);ax.legend()
fig.tight_layout();fig.savefig(review/'three_signals_overlay.png',dpi=240);plt.close(fig)
print(summary.to_string(index=False));print('VALID',len(valid),'TOTAL',len(rows))


