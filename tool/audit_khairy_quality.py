from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'Data/Exprement Data/Khairy'
OUT=ROOT/'Report/khairy_quality_review'
OUT.mkdir(parents=True,exist_ok=True)
RUN=BASE/'Result/generalization_20261016_112916_799339'
split=pd.read_csv(RUN/'data_split.csv').set_index('trial').split.to_dict()
metrics=pd.concat([pd.read_csv(RUN/'test_trial_metrics.csv'),pd.read_csv(RUN/'generalization_trial_metrics.csv')])
records=[];trials={}
for size in ['3','3.5','4']:
 for trial in sorted((BASE/size).glob('TRIAL_*')):
  c=pd.read_csv(trial/'camera.csv');d=pd.read_csv(trial/'grasp_data_smoothed.csv');m=pd.read_csv(trial/'imu.csv').sort_values('t_imu').drop_duplicates('t_imu',keep='last')
  y=c.finger_width.to_numpy();a=c.object_area.to_numpy();ys=pd.Series(y).rolling(3,center=True,min_periods=1).median().to_numpy();ar=pd.Series(a).rolling(3,center=True,min_periods=1).median().to_numpy()
  peak=int(np.argmax(ys));base=float(np.median(ar[:min(10,peak)]));dip=int(np.argmin(ar[:peak+1]));pred=metrics.loc[metrics.trial==trial.name] if 'trial' in metrics else pd.DataFrame()
  row=dict(trial=trial.name,size_cm=float(size),split=split.get(trial.name,'not in run'),rows=len(c),duration=c.t_camera.iloc[-1]-c.t_camera.iloc[0],initial_fw=np.median(y[:5]),peak_fw=ys[peak],peak_progress=100*peak/(len(c)-1),oa_baseline=base,oa_min_ratio=ar[dip]/base,oa_dip_frame=int(c.frame.iloc[dip]),oa_initial_step=np.max(np.abs(np.diff(ar)))/base,fw_raw_max_jump=np.max(np.abs(np.diff(y))),oa_tail_relative_range=np.ptp(ar[-16:])/np.median(ar[-16:]),fw_tail_range=np.ptp(ys[-16:]),initial_abs_acc=np.median(np.abs(d.forward_raw.iloc[:5])),imu_gap=np.diff(m.t_imu).max(),camera_gap=np.diff(c.t_camera).max())
  records.append(row);trials[trial.name]=(trial,c,d)
audit=pd.DataFrame(records);audit.to_csv(OUT/'all_trial_quality.csv',index=False)
print(audit.groupby('size_cm').size().to_string())
suspect=audit.loc[(audit.oa_min_ratio<.75)|(audit.fw_raw_max_jump>70)|(audit.oa_tail_relative_range>.25)|(audit.fw_tail_range>25)|(audit.initial_fw>60)|(audit.peak_progress<35)|(audit.rows<45)].copy()
suspect=suspect.sort_values(['size_cm','oa_min_ratio'])
suspect.to_csv(OUT/'review_candidates.csv',index=False)
print(suspect[['trial','split','rows','oa_min_ratio','fw_raw_max_jump','initial_fw','peak_progress','fw_tail_range']].to_string(index=False),flush=True)
for page in range(0,len(suspect),6):
 rows=suspect.iloc[page:page+6];fig,axs=plt.subplots(len(rows),5,figsize=(18,2.6*len(rows)),squeeze=False)
 for pos,r in enumerate(rows.itertuples()):
  trial,c,d=trials[r.trial];x=c.frame.to_numpy();peak=int(np.argmax(c.finger_width));dip=int(np.where(x==r.oa_dip_frame)[0][0])
  for ax,col in zip(axs[pos,:2],['finger_width','object_area']):
   ax.plot(x,c[col],alpha=.6);ax.plot(x,d[col],lw=1.5);ax.set_title(f'{r.trial.rsplit("_",1)[1]} {col}',fontsize=9);ax.grid(alpha=.2)
  for ax,k in zip(axs[pos,2:],[0,dip,peak]):
   fp=trial/'frames'/f'frame_{int(x[k]):06d}.jpg';im=cv2.imdecode(np.fromfile(fp,np.uint8),1)
   ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));ax.set_title(f'{r.size_cm:g}cm frame {x[k]}',fontsize=9);ax.axis('off')
 fig.tight_layout();fig.savefig(OUT/f'review_{page//6+1}.jpg',dpi=110);plt.close(fig)
