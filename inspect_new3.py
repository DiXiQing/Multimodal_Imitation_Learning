from pathlib import Path
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
b=Path('Data/BackupData/D')
for n in ['101653','101833','102404','102801','102900','103837','103941']:
 p=next(b.glob('*20261012_'+n));d=pd.read_csv(p/'camera.csv');y=d.finger_width.rolling(5,center=True,min_periods=1).median().to_numpy();peaks,pr=find_peaks(y,prominence=20,distance=20)
 print(n,len(d),'peaks',[(int(k),round(float(y[k]),1),round(float(v),1)) for k,v in zip(peaks,pr['prominences'])]);print(list(zip(d.frame.to_numpy()[::20],np.round(y[::20],1))))
