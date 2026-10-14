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
from scipy.signal import find_peaks
source=Path(r'C:\MineApp\Code\Multimodal_Imitation_Learning\Data\BackupData\D\TRIAL_Size_3.5cm_20261014_195420')
out=source/'grasp_segments'
out.mkdir(exist_ok=True)
c=pd.read_csv(source/'camera.csv'); m=pd.read_csv(source/'imu.csv')
y=c.finger_width.rolling(5,center=True,min_periods=1).median().to_numpy()
p,props=find_peaks(y,prominence=60,distance=50)
print('camera',len(c),'duration',c.t_camera.iloc[-1]-c.t_camera.iloc[0],'imu',len(m),'columns',m.columns.tolist())
print('camera dt',np.quantile(np.diff(c.t_camera),[0,.5,.95,1]),'IMU dt',np.quantile(np.diff(m.t_imu),[0,.5,.95,1]))
print('coverage',m.t_imu.iloc[0]-c.t_camera.iloc[0],m.t_imu.iloc[-1]-c.t_camera.iloc[-1],'duplicates',m.t_imu.duplicated().sum(),'index gaps',m.imu_index.diff().value_counts().head().to_dict())
print(pd.DataFrame({'peak':p,'fw':y[p],'oa':c.object_area.iloc[p].to_numpy(),'prom':props['prominences']}).to_string(index=False))
fig,axs=plt.subplots(3,1,figsize=(22,10),sharex=True)
for ax,col in zip(axs,['finger_width','object_area']):
 ax.plot(c.frame,c[col],lw=.7)
 for k in p: ax.axvline(k,alpha=.2,color='r'); ax.text(k,c[col].max(),str(k),rotation=90,fontsize=6)
 ax.set_ylabel(col);ax.grid(alpha=.2)
axs[2].plot(np.interp(m.t_imu,c.t_camera,c.frame),m.forward_raw,lw=.5);axs[2].set_ylabel('forward_raw');fig.tight_layout();fig.savefig(out/'recording_overview.png',dpi=120)
for page in range(0,len(p),12):
 chosen=p[page:page+12];fig,axs=plt.subplots(len(chosen),3,figsize=(12,2*len(chosen)),squeeze=False)
 for row,k in enumerate(chosen):
  for ax,f in zip(axs[row],[max(0,k-35),k,min(len(c)-1,k+40)]):
   im=cv2.imdecode(np.fromfile(source/'frames'/f'frame_{int(c.frame.iloc[f]):06d}.jpg',np.uint8),1)
   ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));ax.set_title(f'peak {k} / frame {f}',fontsize=9);ax.axis('off')
 fig.tight_layout();fig.savefig(out/f'candidates_{page//12+1}.jpg',dpi=100);plt.close(fig)

