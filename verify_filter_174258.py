import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import butter,sosfilt,sosfilt_zi
s=Path(r'C:\MineApp\Code\Multimodal_Imitation_Learning\Data\BackupData\D\TRIAL_Size_2.5cm_20261014_174258\grasp_segments')
for name in ['TRIAL_Size_2.5cm_20261014_174301','TRIAL_Size_3cm_20261014_174652']:
 p=s/name
 if not (p/'grasp_data_smoothed.csv').exists():continue
 raw=pd.read_csv(p/'grasp_data.csv');sm=pd.read_csv(p/'grasp_data_smoothed.csv');imu=pd.read_csv(p/'imu.csv').sort_values('t_imu',kind='stable').drop_duplicates('t_imu',keep='last');ti=imu.t_imu.to_numpy();tc=raw.t_camera.to_numpy();grid=ti[0]+np.arange(int(np.ceil((tc[-1]-ti[0])*50))+1)/50
 u=np.interp(grid,ti,imu.forward_raw);sos=butter(2,3,fs=50,output='sos');v=sosfilt(sos,u,zi=sosfilt_zi(sos)*u[0])[0];print("max CSV roundtrip difference",np.max(np.abs(sm.forward_acceleration-np.interp(tc,grid,v))));assert np.allclose(sm.forward_acceleration,np.interp(tc,grid,v),atol=2e-5)
 for col in ['finger_width','object_area']:assert np.allclose(sm[col],raw[col].rolling(3,min_periods=1).median().ewm(span=5,adjust=False).mean())
 assert np.array_equal(raw.forward_raw,sm.forward_raw)
 print(name,'independent filter reconstruction passed')

