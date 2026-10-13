from pathlib import Path
import sys
import json
import hashlib
import shutil
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Analyze import calculate_3d_speed
from Smooth import smooth_trial

source = ROOT / 'Data/BackupData/D/TRIAL_Size_3.5cm_20261013_150004_279664'
output = source / 'grasp_segments'
ranges = [(235,305), (388,459), (569,635), (782,847), (1028,1093), (1205,1274)]
originals = [source / n for n in ['camera.csv','imu.csv','recorded_video.mp4','imu_processing.json','trial_summary.json']]
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
hashes = {p.name: digest(p) for p in originals}
camera = pd.read_csv(source/'camera.csv').sort_values('frame')
imu = pd.read_csv(source/'imu.csv')
ordered = imu.sort_values('t_imu').drop_duplicates('t_imu',keep='last')
output.mkdir(exist_ok=True)
curves = []
records = []
for number, (start,end) in enumerate(ranges,1):
    destination = output / f'{source.name}_segment{number:02d}'
    if destination.exists():
        raise RuntimeError(f'Refusing to overwrite {destination}')
    d = camera.loc[camera.frame.between(start,end)].copy()
    assert len(d) == end-start+1
    tc = d.t_camera.to_numpy()
    # Keep untouched raw samples, including one supporting sample on either
    # side of camera boundaries for exact interpolation coverage.
    before = ordered.loc[ordered.t_imu<=tc[0]].tail(1)
    after = ordered.loc[ordered.t_imu>=tc[-1]].head(1)
    if before.empty or after.empty:
        raise RuntimeError('IMU does not cover the segment')
    t0,t1 = before.t_imu.iloc[0],after.t_imu.iloc[0]
    raw_imu = imu.loc[imu.t_imu.between(t0,t1)].copy()
    segment = ordered.loc[ordered.t_imu.between(t0,t1)]
    ti = segment.t_imu.to_numpy()
    axes = segment[['ax','ay','az']].to_numpy()
    _,_,_,speed = calculate_3d_speed(ti,axes)
    for j,col in enumerate(['ax','ay','az']):
        d[col] = np.interp(tc,ti,axes[:,j])
    d['mag'] = np.linalg.norm(d[['ax','ay','az']],axis=1)
    d['velocity'] = np.interp(tc,ti,speed)
    smooth = smooth_trial(d)
    smooth['mag'] = np.linalg.norm(smooth[['ax','ay','az']],axis=1)
    destination.mkdir()
    camera.loc[camera.frame.between(start,end)].to_csv(destination/'camera.csv',index=False)
    raw_imu.to_csv(destination/'imu.csv',index=False)
    d.to_csv(destination/'grasp_data.csv',index=False)
    smooth.to_csv(destination/'grasp_data_smoothed.csv',index=False,float_format='%.9g')
    shutil.copy2(source/'imu_processing.json',destination/'imu_processing.json')
    frame_dir = destination/'frames'
    frame_dir.mkdir()
    video_temp = ROOT / 'analysis' / f'150004_segment{number:02d}_temp.mp4'
    writer = None
    try:
        for frame in d.frame.astype(int):
            image_path = source/'frames'/f'frame_{frame:06d}.jpg'
            img = cv2.imdecode(np.fromfile(image_path,dtype=np.uint8),cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f'Cannot read {image_path}')
            if writer is None:
                height,width = img.shape[:2]
                writer = cv2.VideoWriter(str(video_temp),cv2.VideoWriter_fourcc(*'mp4v'),30,(width,height))
                if not writer.isOpened():
                    raise RuntimeError('Video writer failed')
            writer.write(img)
            shutil.copy2(image_path,frame_dir/image_path.name)
    finally:
        if writer is not None:
            writer.release()
    cap = cv2.VideoCapture(str(video_temp))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert count == len(d), (count,len(d))
    shutil.move(str(video_temp),str(destination/'grasp_video.mp4'))
    rec = dict(segment=number,source_trial=source.name,frame_start=start,frame_end=end,
               camera_frames=len(d),imu_samples=len(raw_imu),duration_seconds=float(tc[-1]-tc[0]),
               imu_max_gap_seconds=float(np.diff(ti).max()),
               fw_zero_frames=int((d.finger_width<=0).sum()),oa_zero_frames=int((d.object_area<=0).sum()),
               frame_numbering='Original source frame numbers retained',
               video_fps=30,source_sha256=hashes)
    (destination/'crop_metadata.json').write_text(json.dumps(rec,indent=2),encoding='utf-8')
    fig,axs = plt.subplots(3,1,figsize=(11,8),sharex=True)
    for ax,col,label in zip(axs,['finger_width','object_area','mag'],['Finger width (px)','Object area (px2)','Acceleration (m/s2)']):
        ax.plot(d.frame,d[col],alpha=.4,label='Raw')
        ax.plot(smooth.frame,smooth[col],label='Smoothed')
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axs[0].legend()
    axs[-1].set_xlabel('Source frame')
    fig.tight_layout()
    fig.savefig(destination/'smoothing_preview.png',dpi=140)
    plt.close(fig)
    records.append(rec)
    curves.append(smooth)
    print(number,start,end,len(d),'frames',len(raw_imu),'IMU', 'gap',rec['imu_max_gap_seconds'],flush=True)

fig,axs = plt.subplots(2,2,figsize=(15,8))
signal_axes = [axs[0,0],axs[0,1],axs[1,0]]
for number,d in enumerate(curves,1):
    progress = 100*(d.frame-d.frame.iloc[0])/(d.frame.iloc[-1]-d.frame.iloc[0])
    for ax,col in zip(signal_axes,['finger_width','object_area','mag']):
        ax.plot(progress,d[col],label=f'Grasp {number}',color=plt.cm.tab10(number-1))
for ax,label in zip(signal_axes,['Finger width (px)','Object area (px2)','Acceleration magnitude (m/s2)']):
    ax.set(xlabel='Recorded grasp progress (%)',ylabel=label,xlim=(0,100))
    ax.grid(alpha=.2)
axs[1,1].axis('off')
h,l = signal_axes[0].get_legend_handles_labels()
axs[1,1].legend(h,l,loc='center',frameon=False,title='6 grasps / 3.5 cm')
fig.tight_layout()
fig.savefig(output/'six_grasps_smooth_overlay.png',dpi=150)
plt.close(fig)
pd.DataFrame([{k:v for k,v in r.items() if k!='source_sha256'} for r in records]).to_csv(output/'segment_summary.csv',index=False)
assert hashes == {p.name:digest(p) for p in originals}
print('Verified: all 6 videos match frame counts; original files unchanged.')
