from pathlib import Path
from datetime import datetime, timezone, timedelta
import sys
import hashlib
import shutil
import argparse
import re
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import cv2
from scipy.signal import find_peaks, butter, sosfilt, sosfilt_zi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from Smooth import smooth_trial, median_then_ema
BASE=ROOT/'Data/Exprement Data/Khairy'
OUT=BASE/'grasp_segments'


def file_digest(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.digest()


def read_image(source,frame):
    path=source/'frames'/f'frame_{int(frame):06d}.jpg'
    image=cv2.imdecode(np.fromfile(path,np.uint8),1)
    if image is None:raise ValueError(f'Cannot read {path}')
    return image


def find_ranges(source,c,imu):
    y=c.finger_width.rolling(5,center=True,min_periods=1).median().to_numpy()
    oa=c.object_area.rolling(3,center=True,min_periods=1).median().to_numpy()
    peaks,props=find_peaks(y,prominence=35,distance=20)
    records=[]
    for peak,prom in zip(peaks,props['prominences']):
        r=dict(source=source.name,size_cm=float(re.search(r'Size_([\d.]+)cm',source.name)[1]),peak=int(peak),status='excluded',reason='')
        look=max(0,peak-90)
        low=float(np.quantile(y[look:peak],.15));height=y[peak]-low
        if height<45:r['reason']='small adjustment';records.append(r);continue
        threshold=low+.15*height
        below=np.flatnonzero(y[look:peak]<threshold)
        if not len(below):r['reason']='no resting onset';records.append(r);continue
        onset=look+int(below[-1])+1
        if peak-onset>45:r['reason']='release/return peak after an earlier opening';records.append(r);continue
        start=max(0,onset-25)
        while start<onset-8 and (np.max(y[start:start+5])>threshold or np.ptp(oa[start:start+5])/max(np.median(oa[start:start+5]),1)>.15):start+=1
        r.update(start=start,onset=onset,pre_frames=onset-start)
        if np.median(y[start:onset])>75 or np.ptp(y[start:onset])>45:
            r['reason']='no quiet pre-grasp interval';records.append(r);continue
        baseline=float(np.median(oa[start:onset]));stable=None
        for j in range(peak+5,min(peak+60,len(c)-16)):
            fw=y[j:j+16];a=oa[j:j+16]
            if low+.12*height<=np.median(fw)<=low+.70*height and np.ptp(fw)<=max(18,.10*y[peak]) and np.ptp(a)/max(np.median(a),1)<=.18 and np.median(a)/max(baseline,1)>=1.22:
                stable=j;break
        if stable is None:r['reason']='no stable 15-frame grasp tail with OA increase';records.append(r);continue
        end=stable+15
        r.update(stable=stable,end=end,area_gain=float(np.median(oa[stable:end+1])/max(baseline,1)))
        if any(abs(onset-a['onset'])<35 or start<=a['end'] for a in records if a['status']=='accepted'):
            r['reason']='duplicate opening / overlapping cycle';records.append(r);continue
        d=c.iloc[start:end+1];tc=d.t_camera.to_numpy()
        before=imu.loc[imu.t_imu<=tc[0]].tail(1);after=imu.loc[imu.t_imu>=tc[-1]].head(1)
        if before.empty or after.empty:r['reason']='IMU coverage'
        elif (d.finger_width<=0).any() or (d.object_area<=0).any():r['reason']='FW/OA detection lost'
        elif np.diff(tc).max()>.1:r['reason']='camera gap >100ms'
        else:
            selected=imu.loc[imu.t_imu.between(before.t_imu.iloc[0],after.t_imu.iloc[0])]
            r['imu_max_gap_seconds']=float(np.diff(selected.t_imu).max())
            if r['imu_max_gap_seconds']>.25:r['reason']='IMU gap >250ms'
            else:r['status']='accepted'
        records.append(r)
    return records


def contact_sheets(source,c,records):
    size=re.search(r'Size_([\d.]+)cm',source.name)[1]
    chosen=[r for r in records if r['status']=='accepted']
    for page in range(0,len(chosen),8):
        rows=chosen[page:page+8];fig,axs=plt.subplots(len(rows),4,figsize=(14,2.3*len(rows)),squeeze=False)
        for row,r in enumerate(rows):
            for ax,key in zip(axs[row],['start','peak','stable','end']):
                f=int(c.frame.iloc[int(r[key])]);im=read_image(source,f)
                ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));ax.set_title(f'P{r["peak"]} {key} {f}',fontsize=9);ax.axis('off')
        fig.tight_layout();fig.savefig(OUT/f'accepted_{size}cm_{page//8+1}.jpg',dpi=90);plt.close(fig)


def export(source,c,original_imu,imu,r):
    d=c.iloc[r['start']:r['end']+1].copy();tc=d.t_camera.to_numpy()
    before=imu.loc[imu.t_imu<=tc[0]].tail(1).t_imu.iloc[0]
    grid=before+np.arange(int(np.ceil((tc[-1]-before)*50))+1)/50
    last=imu.loc[imu.t_imu>=grid[-1]].head(1)
    if last.empty:raise ValueError('50 Hz grid not covered by IMU')
    support=imu.loc[imu.t_imu.between(before,last.t_imu.iloc[0])]
    ti=support.t_imu.to_numpy()
    if np.diff(ti).max()>.25:raise ValueError('IMU grid support gap')
    for col in ['ax','ay','az','forward_raw','velocity']:
        if col in support:d[col]=np.interp(tc,ti,support[col])
    d['mag']=np.linalg.norm(d[['ax','ay','az']],axis=1)
    sos=butter(2,3,fs=50,output='sos');filtered={}
    for col in ['ax','ay','az','forward_raw']:
        u=np.interp(grid,ti,support[col]);f=sosfilt(sos,u,zi=sosfilt_zi(sos)*u[0])[0]
        filtered[col]=np.interp(tc,grid,f)
    smooth=smooth_trial(d,np.column_stack([filtered[k] for k in ['ax','ay','az']]))
    smooth['forward_acceleration']=filtered['forward_raw']
    stamp=datetime.fromtimestamp(float(tc[0]),timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')
    dest=OUT/f'TRIAL_Size_{r["size_cm"]:g}cm_{stamp}'
    if dest.exists():raise ValueError(f'Already exists: {dest}')
    dest.mkdir();(dest/'frames').mkdir()
    d.to_csv(dest/'grasp_data.csv',index=False);smooth.to_csv(dest/'grasp_data_smoothed.csv',index=False)
    c.iloc[r['start']:r['end']+1].to_csv(dest/'camera.csv',index=False)
    original_imu.loc[original_imu.t_imu.between(before,last.t_imu.iloc[0])].to_csv(dest/'imu.csv',index=False)
    writer=None
    for frame in d.frame.astype(int):
        im=read_image(source,frame);fp=source/'frames'/f'frame_{frame:06d}.jpg'
        if writer is None:
            h,w=im.shape[:2];writer=cv2.VideoWriter(str(dest/'grasp_video.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(w,h))
            if not writer.isOpened():raise ValueError('Cannot create video')
        writer.write(im);shutil.copy2(fp,dest/'frames'/fp.name)
        assert hashlib.sha256(fp.read_bytes()).digest()==hashlib.sha256((dest/'frames'/fp.name).read_bytes()).digest()
    writer.release();cap=cv2.VideoCapture(str(dest/'grasp_video.mp4'));count=0
    while True:
        ok,_=cap.read()
        if not ok:break
        count+=1
    cap.release();assert count==len(d)==len(list((dest/'frames').glob('*.jpg')))
    for name in ['camera.csv','grasp_data.csv','grasp_data_smoothed.csv']:
        check=pd.read_csv(dest/name);assert np.array_equal(check.frame,d.frame);assert np.allclose(check.t_camera,tc,rtol=0,atol=1e-6)
    assert np.array_equal(smooth.forward_raw,d.forward_raw)
    for col in ['finger_width','object_area']:assert np.allclose(smooth[col],median_then_ema(d[col]))
    assert np.isfinite(smooth.select_dtypes(include='number')).all().all()
    summary=dict(**r,trial=dest.name,rows=len(d),video_frames=count,frame_start=int(d.frame.iloc[0]),frame_end=int(d.frame.iloc[-1]),imu_rows=len(support),imu_max_gap_with_support=float(np.diff(ti).max()),imu_gaps_over_60ms=int((np.diff(ti)>.06).sum()),camera_max_gap=float(np.diff(tc).max()))
    print('Saved',dest.name,len(d),flush=True)
    return summary,smooth


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true');parser.add_argument('--no-sheets',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True);all_records=[];inputs=[];audits=[]
    for source in sorted(BASE.glob('TRIAL_*')):
        c=pd.read_csv(source/'camera.csv');original=pd.read_csv(source/'imu.csv')
        imu=original.sort_values('t_imu',kind='stable').drop_duplicates('t_imu',keep='last')
        assert np.isfinite(c[['frame','t_camera','finger_width','object_area']]).all().all()
        assert (np.diff(c.frame)>0).all() and (np.diff(c.t_camera)>0).all()
        assert np.isfinite(imu[['t_imu','ax','ay','az','forward_raw']]).all().all()
        records=find_ranges(source,c,imu);all_records.extend(records)
        if not args.no_sheets:contact_sheets(source,c,records)
        inputs.append((source,c,original,imu,records))
        audits.append(dict(source=source.name,camera_rows=len(c),imu_rows=len(original),duplicate_imu_times=int(original.t_imu.duplicated().sum()),backward_imu_times=int((original.t_imu.diff()<0).sum()),imu_index_gaps=int((original.imu_index.diff()>1).sum()),imu_max_gap_seconds=float(np.diff(imu.t_imu).max()),imu_gaps_over_60ms=int((np.diff(imu.t_imu)>.06).sum()),camera_max_gap_seconds=float(np.diff(c.t_camera).max()),fw_zero=int((c.finger_width<=0).sum()),oa_zero=int((c.object_area<=0).sum())))
    ranges=pd.DataFrame(all_records);ranges.to_csv(OUT/'crop_ranges.csv',index=False)
    pd.DataFrame(audits).to_csv(OUT/'recording_audit.csv',index=False)
    print(ranges.groupby(['size_cm','status']).size().to_string(),flush=True)
    print(ranges.query('status == "accepted"')[['size_cm','peak','start','stable','end','pre_frames']].to_string(index=False),flush=True)
    if not args.export:return
    paths=[p for source,*_ in inputs for p in source.rglob('*') if p.is_file()]
    print('Hashing original recording files:',len(paths),flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:originals=dict(zip(paths,pool.map(file_digest,paths)))
    print('Original hashes captured; exporting',flush=True)
    summaries=[];curves=[]
    for source,c,original,imu,records in inputs:
        for r in records:
            if r['status']=='accepted':
                summary,smooth=export(source,c,original,imu,r);summaries.append(summary);curves.append((r,smooth))
    pd.DataFrame(summaries).to_csv(OUT/'crop_summary.csv',index=False)
    for size in [None,3,3.5,4]:
        selected=[(r,d) for r,d in curves if size is None or r['size_cm']==size]
        fig,axs=plt.subplots(3,1,figsize=(13,10),sharex=True)
        for i,(r,d) in enumerate(selected):
            x=100*(d.t_camera-d.t_camera.iloc[0])/(d.t_camera.iloc[-1]-d.t_camera.iloc[0])
            color={3:'#2563eb',3.5:'#16a34a',4:'#ea580c'}[r['size_cm']] if size is None else plt.cm.turbo(i/max(1,len(selected)-1))
            for ax,col in zip(axs,['finger_width','object_area','forward_acceleration']):ax.plot(x,d[col],color=color,lw=.9,alpha=.5)
        for ax,label in zip(axs,['FW (px)','OA (px²)','Signed forward acceleration (m/s²)']):ax.set_ylabel(label);ax.grid(alpha=.2)
        axs[2].axhline(0,color='black',lw=.7);axs[2].set_xlabel('Segment elapsed time (%)')
        fig.suptitle(f'Khairy | {len(selected)} complete grasps | '+('3 cm blue / 3.5 cm green / 4 cm orange' if size is None else f'{size:g} cm'))
        fig.tight_layout();fig.savefig(OUT/('combined_overlay.png' if size is None else f'overlay_{size:g}cm.png'),dpi=150);plt.close(fig)
    print('Checking original hashes',flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:assert all(digest==originals[p] for p,digest in zip(paths,pool.map(file_digest,paths)))
    (OUT/'verification_report.md').write_text('Khairy cropping completed. Source files including JPEGs were verified unchanged by SHA-256. Every copied JPEG hash, fully decoded video frame count, CSV frame IDs and timestamps checked. See crop_ranges.csv for all decisions and recording_audit.csv for timing quality. FW/OA: trailing 3-frame median + EMA span 5. Acceleration: 50 Hz grid, causal second-order 3 Hz Butterworth, steady-state first-sample initialization, interpolated to camera timestamps. forward_raw is retained unchanged between synchronized raw and smoothed CSV; forward_acceleration contains filtered signed data. No magnitude substitution, shifts, or amplitude adjustment. All accepted intervals have bracketing IMU support and no gaps over 250 ms. Timestamp gaps cannot uniquely distinguish sensor packet loss from notification timing. No JSON generated.\n',encoding='utf-8')
    print('Verified',len(summaries),'segments; original files unchanged',flush=True)


if __name__=='__main__':main()
