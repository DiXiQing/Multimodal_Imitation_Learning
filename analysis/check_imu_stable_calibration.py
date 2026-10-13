from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
import IMUTest
from imu_processing import gravity_in_sensor

def feed(state,acc,angles,t,gyro=(0,0,0)):
    state.process(dict(acc=np.asarray(acc,dtype=float),angles=np.asarray(angles,dtype=float),gyro=np.asarray(gyro,dtype=float)),t)

s=IMUTest.State();s.begin()
for t in np.arange(0,2.02,.02):feed(s,[.11,.14,1.02],[0,0,0],t)
assert not s.calibration_quality()[0], 'Large gravity mismatch must not become a bias'
s.begin()
for t in np.arange(0,2.02,.02):feed(s,gravity_in_sensor([15,-10,30])+[.01,-.01,.02],[15,-10,30],t)
assert s.calibration_quality()[0]
s.finish();feed(s,gravity_in_sensor([45,20,60])+[.01,-.01,.02],[45,20,60],3)
assert s.linear_mag<1e-10
feed(s,gravity_in_sensor([45,20,60])+[.11,-.01,.02],[45,20,60],3.02)
np.testing.assert_allclose(s.linear_mag,.981,atol=1e-10)

source=root/'Data'/'BackupData'/'D'/'TRIAL_Size_4.5cm_20261013_103426'
d=pd.read_csv(source/'imu.csv').sort_values('t_imu')
t=d.t_imu-d.t_imu.iloc[0]
reference=d.loc[t.between(2,5)]
s=IMUTest.State();s.begin()
for row in reference.itertuples():
    feed(s,[row.ax_raw_g,row.ay_raw_g,row.az_raw_g],[row.roll_deg,row.pitch_deg,row.yaw_deg],row.t_imu,[row.gx_dps,row.gy_dps,row.gz_dps])
good,reason=s.calibration_quality()
print('Recorded later window quality:',good,reason)
assert good
s.finish()
raw=d[['ax_raw_g','ay_raw_g','az_raw_g']].to_numpy()
gravity=d[['gravity_x_g','gravity_y_g','gravity_z_g']].to_numpy()
replay=np.linalg.norm((raw-gravity-s.linear_bias)*9.81,axis=1)
mask=t.between(5,10)
print('5-10 s magnitude mean: old',d.loc[mask,'mag'].mean(),'recalibrated replay',replay[mask].mean())
fig,ax=plt.subplots(figsize=(13,5))
ax.plot(t,d.mag,lw=1,label='Recorded (early calibration)')
ax.plot(t,replay,lw=1,label='Diagnostic replay (later stable calibration)')
ax.axvspan(2,5,color='gray',alpha=.15,label='Candidate stable calibration interval')
ax.set(xlabel='Time (s)',ylabel='Acceleration magnitude (m/s²)');ax.grid(alpha=.2);ax.legend();fig.tight_layout()
fig.savefig(root/'analysis'/'103426_calibration_diagnosis.png',dpi=150);plt.close(fig)
print('PASS: bad calibration rejected, stable calibration accepted, tilt cancellation and real movement preserved. Source CSV not modified.')
