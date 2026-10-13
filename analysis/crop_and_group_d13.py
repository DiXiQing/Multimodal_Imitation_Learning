from pathlib import Path
import sys,json,hashlib,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from Analyze import calculate_3d_speed
from Smooth import smooth_trial
base=root/'Data'/'临时数据'/'D13';review=base/'review';review.mkdir(exist_ok=True)
rows=[];curves={};review_items=[]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for trial in sorted(base.glob('TRIAL_*')):
    source={p.name:digest(p) for p in [trial/'camera.csv',trial/'imu.csv'] if p.exists()}
    c=pd.read_csv(trial/'camera.csv');im0=pd.read_csv(trial/'imu.csv')
    meta=json.loads((trial/'imu_processing.json').read_text()) if (trial/'imu_processing.json').exists() else {}
    method='stable_calibration' if meta.get('calibration_quality')=='Stable calibration samples' else 'early_calibration'
    rec=dict(trial=trial.name,recording_version=method,source_camera_rows=len(c),source_imu_rows=len(im0),status='pending',flags=[])
    if not len(c) or not len(im0):
        rec.update(group='04_failed_recordings',status='failed',reason='Empty camera or IMU CSV')
        rows.append(rec);continue
    long=len(c)>600
    rec['group']='03_multi_action_review' if long else ('01_stable_calibration' if method=='stable_calibration' else '02_early_calibration')
    try:
        c=c.sort_values('frame').reset_index(drop=True)
        im=im0.sort_values('t_imu').drop_duplicates('t_imu',keep='last')
        y=c.finger_width.rolling(5,center=True,min_periods=1).median().to_numpy()
        peaks,props=find_peaks(y,prominence=60,distance=25)
        qualifying=[int(k) for k in peaks if y[k]>=max(180,.7*np.nanmax(y))]
        if not qualifying:raise ValueError('No substantial complete aperture peak')
        candidates=[]
        for peak in qualifying:
            look=max(0,peak-100)
            low=float(np.nanmin(y[look:peak+1]));threshold=low+.1*(y[peak]-low)
            below=np.flatnonzero(y[look:peak]<threshold)
            if not len(below):continue
            onset=look+int(below[-1])+1;start=max(0,onset-15)
            stable=None
            # First stable closed grip; retain 15 more real frames.
            for j in range(peak+8,min(len(y)-8,peak+95)):
                window=y[j:j+9];level=np.median(window)
                if low+.2*(y[peak]-low)<=level<=low+.7*(y[peak]-low) and np.ptp(window)<=max(10,.04*y[peak]):
                    stable=j;break
            if stable is None:
                closing=np.flatnonzero(y[peak+1:min(len(y),peak+95)]<=low+.5*(y[peak]-low))
                if not len(closing):continue
                stable=peak+1+int(closing[0])
            end=min(len(c)-1,stable+15)
            prearea=float(np.median(c.object_area.iloc[start:onset+1]));postarea=float(np.median(c.object_area.iloc[stable:end+1]))
            gain=postarea/max(prearea,1)
            if gain<1.25:continue
            candidates.append((start,end,peak,stable,gain))
        if not candidates:raise ValueError('No aperture peak followed by closing and target-area growth')
        start,end,peak,stable,gain=candidates[0]
        if long:rec['flags'].append(f'Long recording: {len(candidates)} grasp-like candidates; first complete candidate selected')
        d=c.iloc[start:end+1].copy();tc=d.t_camera.to_numpy()
        segment=im.loc[im.t_imu.between(tc[0],tc[-1])]
        if len(segment)<10:raise ValueError('Insufficient overlapping IMU')
        ti=segment.t_imu.to_numpy();axes=segment[['ax','ay','az']].to_numpy()
        if (ti[-1]-ti[0])/(tc[-1]-tc[0])<.9:raise ValueError('IMU does not cover selected action')
        if np.diff(ti).max()>.25:rec['flags'].append('IMU gap exceeds 0.25 seconds')
        _,_,_,speed=calculate_3d_speed(ti,axes)
        for j,col in enumerate(['ax','ay','az']):d[col]=np.interp(tc,ti,axes[:,j])
        d['mag']=np.linalg.norm(d[['ax','ay','az']],axis=1);d['velocity']=np.interp(tc,ti,speed)
        smooth=smooth_trial(d);smooth['mag']=np.linalg.norm(smooth[['ax','ay','az']],axis=1)
        if (d.object_area<=0).any():rec['flags'].append('Object area has zero detections; retained without repair')
        if len(d)>90:rec['flags'].append('Long crop (>90 frames)')
        rec.update(status='cropped',frame_start=int(d.frame.iloc[0]),frame_end=int(d.frame.iloc[-1]),peak_frame=int(c.frame.iloc[peak]),rows=len(d),area_gain=gain,
                   initial_acceleration=float(smooth.mag.iloc[:5].mean()),final_acceleration=float(smooth.mag.iloc[-5:].mean()),acc_peak=float(smooth.mag.max()),fw_peak=float(smooth.finger_width.max()))
        for p in [trial/'grasp_data.csv',trial/'grasp_data_smoothed.csv']:
            if p.exists():raise ValueError('Existing processed CSV found; refusing overwrite')
        d.to_csv(trial/'grasp_data.csv',index=False)
        smooth.to_csv(trial/'grasp_data_smoothed.csv',index=False,float_format='%.9g')
        fig,axs=plt.subplots(3,1,figsize=(11,8),sharex=True)
        for ax,col,label in zip(axs,['finger_width','object_area','mag'],['Finger width (px)','Object area (px²)','Acceleration (m/s²)']):
            ax.plot(d.frame,d[col],alpha=.4,label='Raw');ax.plot(smooth.frame,smooth[col],label='Smoothed');ax.set_ylabel(label);ax.grid(alpha=.2)
        axs[0].legend();axs[-1].set_xlabel('Frame');fig.tight_layout();fig.savefig(trial/'smoothing_preview.png',dpi=140);plt.close(fig)
        curves[trial.name]=smooth
        review_items.append((trial.name,c,y,start,end))
        (trial/'crop_metadata.json').write_text(json.dumps(dict(**rec,source_sha256=source,method='First substantial grasp peak with area growth; 15 frames pre-onset and stable grip plus 15 tail frames; sorted/deduplicated IMU only in memory'),indent=2),encoding='utf-8')
    except Exception as e:
        rec.update(status='review_required',reason=str(e))
    assert source=={name:digest(trial/name) for name in source},trial.name
    rows.append(rec)
    print(trial.name[-6:],rec['group'],rec['status'],rec.get('frame_start'),rec.get('frame_end'),rec.get('reason',''),flush=True)

for page in range(0,len(review_items),6):
    items=review_items[page:page+6];fig,axs=plt.subplots(len(items),1,figsize=(15,2.7*len(items)),squeeze=False)
    for ax,(name,c,y,start,end) in zip(axs[:,0],items):
        ax.plot(c.frame,y,lw=1);ax.axvspan(c.frame.iloc[start],c.frame.iloc[end],color='green',alpha=.2);ax.set_title(name[-6:]);ax.set_ylabel('FW (px)');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(review/f'crop_review_{page//6+1}.png',dpi=120);plt.close(fig)

grid=np.linspace(0,100,201)
for group in ['01_stable_calibration','02_early_calibration','03_multi_action_review']:
    chosen=[r for r in rows if r['group']==group and r['status']=='cropped']
    if not chosen:continue
    fig,axs=plt.subplots(2,2,figsize=(16,8));signal_axes=[axs[0,0],axs[0,1],axs[1,0]]
    for i,r in enumerate(chosen):
        d=curves[r['trial']];x=100*(d.frame-d.frame.iloc[0])/(d.frame.iloc[-1]-d.frame.iloc[0])
        for ax,col in zip(signal_axes,['finger_width','object_area','mag']):ax.plot(x,d[col],color=plt.cm.tab20(i/max(1,len(chosen)-1)),lw=1.2,label=r['trial'][-6:])
    for ax,label in zip(signal_axes,['Finger width (px)','Object area (px²)','Acceleration magnitude (m/s²)']):ax.set(xlabel='Recorded trial progress (%)',ylabel=label,xlim=(0,100));ax.grid(alpha=.2)
    axs[1,1].axis('off');h,l=signal_axes[0].get_legend_handles_labels();axs[1,1].legend(h,l,loc='center',ncol=3,frameon=False,title=f'{group}: {len(chosen)} trials');fig.tight_layout();fig.savefig(review/(group+'_overlay.png'),dpi=150);plt.close(fig)

pd.DataFrame(rows).to_csv(review/'crop_and_group_summary.csv',index=False,encoding='utf-8-sig')
(review/'grouping_manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
# Reversible organization; no source CSV edits or frame deletions.
for r in rows:
    src=base/r['trial'];folder=base/r['group'];folder.mkdir(exist_ok=True);dst=folder/src.name
    assert src.resolve().parent==base.resolve() and dst.resolve().parent==folder.resolve() and not dst.exists()
    shutil.move(str(src),str(dst))
print('SUMMARY',pd.DataFrame(rows).groupby(['group','status']).size().to_string())
