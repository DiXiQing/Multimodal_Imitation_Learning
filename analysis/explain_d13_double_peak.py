from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
root=Path(__file__).resolve().parents[1]
base=root/'Data'/'临时数据'/'D13'
trial=next((base/'01_stable_calibration').glob('*_105610'))
d=pd.read_csv(trial/'grasp_data_smoothed.csv');out=base/'review'
x=d.frame.to_numpy();first=128;second=147
fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True)
axes[0].plot(x,d.mag,lw=2,color='tab:blue');axes[0].set_ylabel('Acceleration magnitude (m/s²)')
for col in ['ax','ay','az']:axes[1].plot(x,d[col],lw=1.8,label=col)
axes[1].axhline(0,color='black',lw=.8);axes[1].legend();axes[1].set_ylabel('Signed acceleration (m/s²)')
axes[2].plot(x,d.finger_width,lw=2);axes[2].set_ylabel('Finger width (px)');axes[2].set_xlabel('Original frame')
for ax in axes:
    ax.axvline(first,color='tab:orange',ls='--',label='First magnitude peak')
    ax.axvline(second,color='tab:purple',ls='--',label='Second magnitude peak')
    ax.grid(alpha=.2)
axes[0].legend();fig.tight_layout();fig.savefig(out/'105610_double_peak_signals.png',dpi=150);plt.close(fig)
frames=[104,128,135,147,161]
fig,axes=plt.subplots(1,5,figsize=(20,4))
for ax,n in zip(axes,frames):
    ax.imshow(Image.open(trial/'frames'/f'frame_{n:06d}.jpg'));ax.set_title(f'Frame {n}');ax.axis('off')
fig.tight_layout();fig.savefig(out/'105610_double_peak_frames.png',dpi=130);plt.close(fig)
# Confirm sign reversal in a fixed world frame as well as the sensor frame.
imu=pd.read_csv(trial/'imu.csv').sort_values('t_imu').drop_duplicates('t_imu')
for frame in [first,second]:
    row=d.loc[d.frame==frame].iloc[0]
    angle=np.array([np.interp(row.t_camera,imu.t_imu,imu[c]) for c in ['roll_deg','pitch_deg','yaw_deg']])
    r,p,y=np.deg2rad(angle)
    rx=np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
    ry=np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
    rz=np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]])
    world=(rz@ry@rx)@row[['ax','ay','az']].to_numpy(dtype=float)
    print(frame,'world acceleration',world.tolist())
print('Artifacts:',out/'105610_double_peak_signals.png',out/'105610_double_peak_frames.png')
