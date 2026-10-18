from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base=Path(__file__).resolve().parents[1]/'Data/Exprement Data/Khairy'
out=base/'grasp_segments'
out.mkdir(exist_ok=True)
for source in sorted(base.glob('TRIAL_*')):
 c=pd.read_csv(source/'camera.csv');m=pd.read_csv(source/'imu.csv')
 y=c.finger_width.rolling(5,center=True,min_periods=1).median().to_numpy()
 peaks,props=find_peaks(y,prominence=35,distance=40)
 tag=source.name.split('Size_')[1].split('cm_')[0]
 print(source.name,'camera',len(c),'duration',c.t_camera.iloc[-1]-c.t_camera.iloc[0],'IMU',len(m),'cols',m.columns.tolist(),flush=True)
 print('camera gaps',np.quantile(np.diff(c.t_camera),[.5,.95,1]),'imu gaps',np.quantile(np.diff(np.unique(m.t_imu)),[.5,.95,1]),'duplicates',m.t_imu.duplicated().sum(),'reversals',(m.t_imu.diff()<0).sum(),flush=True)
 p=pd.DataFrame(dict(peak=peaks,frame=c.frame.iloc[peaks].to_numpy(),fw=y[peaks],oa=c.object_area.iloc[peaks].to_numpy(),prominence=props['prominences']))
 p.to_csv(out/f'candidates_{tag}cm.csv',index=False);print(p.to_string(index=False),flush=True)
 fig,axs=plt.subplots(3,1,figsize=(22,10),sharex=True)
 for ax,col in zip(axs,['finger_width','object_area']):
  ax.plot(c.frame,c[col],lw=.7);ax.set_ylabel(col)
  for k in peaks:ax.axvline(c.frame.iloc[k],alpha=.2,color='r');ax.text(c.frame.iloc[k],c[col].max(),str(k),rotation=90,fontsize=6)
 axs[2].plot(np.interp(m.t_imu,c.t_camera,c.frame),m.forward_raw,lw=.5);axs[2].set_ylabel('forward_raw');fig.tight_layout();fig.savefig(out/f'overview_{tag}cm.png',dpi=120);plt.close(fig)
 for page in range(0,len(peaks),8):
  chosen=peaks[page:page+8];fig,axs=plt.subplots(len(chosen),3,figsize=(12,2.4*len(chosen)),squeeze=False)
  for row,k in enumerate(chosen):
   for ax,f in zip(axs[row],[max(0,k-30),k,min(len(c)-1,k+30)]):
    im=cv2.imdecode(np.fromfile(source/'frames'/f'frame_{int(c.frame.iloc[f]):06d}.jpg',np.uint8),1)
    ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));ax.set_title(f'peak {k} / frame {int(c.frame.iloc[f])}',fontsize=9);ax.axis('off')
  fig.tight_layout();fig.savefig(out/f'candidates_{tag}cm_{page//8+1}.jpg',dpi=90);plt.close(fig)
