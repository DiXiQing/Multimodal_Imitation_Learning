import sys
import numpy as np
import pandas as pd
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.append(r'C:\Users\Di\AppData\Roaming\Python\Python312\site-packages')
import cv2
from pathlib import Path
from scipy.signal import find_peaks,butter,sosfilt,sosfilt_zi
from datetime import datetime,timezone,timedelta
import hashlib,shutil,argparse
from Smooth import smooth_trial
source=Path(r'C:\MineApp\Code\Multimodal_Imitation_Learning\Data\BackupData\D\TRIAL_Size_2.5cm_20261014_174258')
out=source/'grasp_segments'
c=pd.read_csv(source/'camera.csv');original_imu=pd.read_csv(source/'imu.csv');imu=original_imu.sort_values('t_imu',kind='stable').drop_duplicates('t_imu',keep='last')
y=c.finger_width.rolling(5,center=True,min_periods=1).median().to_numpy();oa=c.object_area.rolling(3,center=True,min_periods=1).median().to_numpy()
peaks,_=find_peaks(y,prominence=60,distance=50)
records=[]
for peak in peaks:
 r=dict(peak=int(peak),status='excluded',reason='')
 look=max(0,peak-65);low=float(np.quantile(y[look:peak],.1));height=y[peak]-low
 below=np.flatnonzero(y[look:peak]<low+.1*height)
 if not len(below):r['reason']='no resting onset';records.append(r);continue
 onset=look+int(below[-1])+1;start=onset-25
 r.update(start=start,onset=onset,size_cm=2.5 if peak<6800 else 3)
 base=np.median(oa[start:onset]);stable=None
 for j in range(peak+8,min(peak+65,len(c)-16)):
  fw=y[j:j+16];a=oa[j:j+16]
  if low+.15*height<=np.median(fw)<=low+.65*height and np.ptp(fw)<=15 and np.ptp(a)/max(np.median(a),1)<=.22 and np.median(a)/max(base,1)>=1.35:
   stable=j;break
 if peak==2760:r['reason']='visually confirmed return/release action'
 elif stable is None:r['reason']='no complete stable grasp for 15 post-stability frames'
 else:
  end=stable+15;r.update(stable=stable,end=end,area_gain=float(np.median(oa[stable:end+1])/max(base,1)))
  d=c.iloc[start:end+1];tc=d.t_camera.to_numpy();before=imu.loc[imu.t_imu<=tc[0]].tail(1);after=imu.loc[imu.t_imu>=tc[-1]].head(1)
  if before.empty or after.empty:r['reason']='IMU coverage'
  elif (d.finger_width<=0).any() or (d.object_area<=0).any():r['reason']='invalid FW/OA tracking'
  elif np.diff(tc).max()>.1:r['reason']='camera gap > 100ms'
  else:
   seg=imu.loc[imu.t_imu.between(before.t_imu.iloc[0],after.t_imu.iloc[0])];gap=float(np.diff(seg.t_imu).max());r['imu_max_gap_seconds']=gap
   if gap>.25:r['reason']='IMU gap > 250ms'
   else:r['status']='accepted'
 records.append(r)
ranges=pd.DataFrame(records);ranges.to_csv(out/'crop_ranges.csv',index=False)
print(ranges.to_string(index=False),flush=True)
chosen=[r for r in records if r['status']=='accepted']
for page in range(0,len(chosen),12):
 rows=chosen[page:page+12];fig,axs=plt.subplots(len(rows),3,figsize=(12,2*len(rows)),squeeze=False)
 for row,r in enumerate(rows):
  for ax,key in zip(axs[row],['start','stable','end']):
   f=r[key];im=cv2.imdecode(np.fromfile(source/'frames'/f'frame_{f:06d}.jpg',np.uint8),1);ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));ax.set_title(f'P{r["peak"]} {key} {f}',fontsize=9);ax.axis('off')
 fig.tight_layout();fig.savefig(out/f'accepted_{page//12+1}.jpg',dpi=90);plt.close(fig)
if '--export' not in sys.argv:sys.exit()
hashes={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in [source/'camera.csv',source/'imu.csv',source/'recorded_video.mp4']}
sos=butter(2,3,fs=50,output='sos');summary=[];curves=[]
for r in chosen:
 d=c.iloc[r['start']:r['end']+1].copy();tc=d.t_camera.to_numpy();before=imu.loc[imu.t_imu<=tc[0]].tail(1).t_imu.iloc[0];after=imu.loc[imu.t_imu>=tc[-1]].head(1).t_imu.iloc[0]
 seg=imu.loc[imu.t_imu.between(before,after)];ti=seg.t_imu.to_numpy()
 stamp=datetime.fromtimestamp(tc[0],timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S');dest=out/f'TRIAL_Size_{r["size_cm"]:g}cm_{stamp}'
 if dest.exists():raise RuntimeError(f'Exists: {dest}')
 dest.mkdir();(dest/'frames').mkdir()
 for col in ['ax','ay','az','velocity','forward_raw']:d[col]=np.interp(tc,ti,seg[col])
 d['mag']=np.linalg.norm(d[['ax','ay','az']],axis=1)
 uniform=ti[0]+np.arange(int(np.ceil((tc[-1]-ti[0])*50))+1)/50
 support=imu.loc[imu.t_imu.between(ti[0],uniform[-1]+.1)]
 if support.t_imu.iloc[-1]<uniform[-1]:raise RuntimeError('Filter grid coverage')
 filtered={}
 for col in ['ax','ay','az','forward_raw']:
  u=np.interp(uniform,support.t_imu,support[col]);f=sosfilt(sos,u,zi=sosfilt_zi(sos)*u[0])[0];filtered[col]=np.interp(tc,uniform,f)
 smooth=smooth_trial(d,np.column_stack([filtered[col] for col in ['ax','ay','az']]));smooth['forward_acceleration']=filtered['forward_raw']
 d.to_csv(dest/'grasp_data.csv',index=False);smooth.to_csv(dest/'grasp_data_smoothed.csv',index=False);c.iloc[r['start']:r['end']+1].to_csv(dest/'camera.csv',index=False)
 original_imu.loc[original_imu.t_imu.between(before,support.t_imu.iloc[-1])].to_csv(dest/'imu.csv',index=False)
 writer=None;frame_hash_ok=True
 for f in d.frame.astype(int):
  fp=source/'frames'/f'frame_{f:06d}.jpg';im=cv2.imdecode(np.fromfile(fp,np.uint8),1)
  if writer is None:
   h,w=im.shape[:2];writer=cv2.VideoWriter(str(dest/'grasp_video.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(w,h))
   assert writer.isOpened()
  writer.write(im);copied=dest/'frames'/fp.name;shutil.copy2(fp,copied);assert hashlib.sha256(fp.read_bytes()).digest()==hashlib.sha256(copied.read_bytes()).digest()
 writer.release();cap=cv2.VideoCapture(str(dest/'grasp_video.mp4'));count=0
 while True:
  ok,im=cap.read()
  if not ok:break
  count+=1
 cap.release();assert count==len(d)==len(list((dest/'frames').glob('*.jpg')))
 for name in ['camera.csv','grasp_data.csv','grasp_data_smoothed.csv']:
  check=pd.read_csv(dest/name);assert np.array_equal(check.frame,d.frame);assert np.allclose(check.t_camera,tc,rtol=0,atol=1e-6)
 assert np.array_equal(smooth.forward_raw,d.forward_raw)
 summary.append(dict(trial=dest.name,**r,rows=len(d),video_frames=count,camera_max_gap_seconds=float(np.diff(tc).max()),imu_samples=len(seg),imu_gaps_over_60ms=int((np.diff(ti)>.06).sum()),start_time=datetime.fromtimestamp(tc[0],timezone(timedelta(hours=9))).isoformat()))
 curves.append((r,smooth));print('Saved',dest.name,len(d),flush=True)
pd.DataFrame(summary).to_csv(out/'crop_summary.csv',index=False)
for size in [None,2.5,3]:
 fig,axs=plt.subplots(3,1,figsize=(14,11),sharex=True)
 selected=[(r,d) for r,d in curves if size is None or r['size_cm']==size]
 for i,(r,d) in enumerate(selected):
  x=100*(d.t_camera-d.t_camera.iloc[0])/(d.t_camera.iloc[-1]-d.t_camera.iloc[0])
  color=plt.cm.Blues(.3+.65*i/max(len(selected)-1,1)) if r['size_cm']==2.5 else plt.cm.Oranges(.3+.65*i/max(len(selected)-1,1))
  for ax,col in zip(axs,['finger_width','object_area','forward_acceleration']):ax.plot(x,d[col],color=color,alpha=.65,lw=1)
 for ax,label in zip(axs,['FW (px)','OA (px2)','Signed forward acceleration (m/s2)']):ax.set_ylabel(label);ax.grid(alpha=.2)
 axs[2].axhline(0,color='black',lw=.7);axs[2].set_xlabel('Segment elapsed time (%)');fig.suptitle(f'{len(selected)} complete grasps | Blue: 2.5 cm, Orange: 3 cm | no mean adjustment');fig.tight_layout();fig.savefig(out/('combined_overlay.png' if size is None else f'overlay_{size:g}cm.png'),dpi=150);plt.close(fig)
assert hashes=={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in hashes}
print('Verified all videos, CSV alignment, copied frames, original hashes. Counts:',pd.DataFrame(summary).groupby('size_cm').size().to_dict())
