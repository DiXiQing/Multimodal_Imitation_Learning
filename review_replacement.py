from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
base=Path('Data/Exprement Data/D11/2')
target=base/'TRIAL_Size_2cm_20261014_094240'
grid=np.linspace(0,100,201)
def curve(p):
    d=pd.read_csv(p/'grasp_data_smoothed.csv');x=100*(d.frame-d.frame.iloc[0])/(d.frame.iloc[-1]-d.frame.iloc[0])
    return np.stack([np.interp(grid,x,np.linalg.norm(d[['ax','ay','az']],axis=1)),np.interp(grid,x,d.finger_width),np.interp(grid,x,d.object_area)])
v=np.stack([curve(p) for p in base.glob('TRIAL_*') if p!=target]);ref=np.median(v,axis=0)
sc=np.percentile(v,95,axis=(0,2))-np.percentile(v,5,axis=(0,2))
rows=[]
for p in (base/'backup').glob('TRIAL_*'):
    d=pd.read_csv(p/'grasp_data_smoothed.csv');im=pd.read_csv(p/'imu.csv').sort_values('t_imu').drop_duplicates('t_imu')
    e=np.sqrt(np.mean((curve(p)-ref)**2,axis=1))
    rows.append(dict(trial=p.name,score=(e/sc)@np.array([.5,.25,.25]),gap=np.diff(im.t_imu).max(),fwpeak=d.finger_width.max(),fwend=d.finger_width.iloc[-1]))
r=pd.DataFrame(rows).sort_values('score');print(r.to_string(index=False));chosen=base/'backup'/'TRIAL_Size_2cm_20261014_094213'
fig,axs=plt.subplots(3,1,figsize=(12,9),sharex=True)
for p,color in [(target,'tab:red'),(chosen,'tab:blue')]:
    for ax,k in zip(axs,range(3)):ax.plot(grid,curve(p)[k],color=color,label=p.name.rsplit('_',1)[1])
for ax,k,lab in zip(axs,range(3),['Acceleration (m/s2)','FW (px)','OA (px2)']):
    ax.plot(grid,ref[k],color='black',ls='--',label='Other 19 median');ax.set_ylabel(lab);ax.legend();ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(base/'replacement_candidate_comparison.png',dpi=140);plt.close(fig)
fig,axs=plt.subplots(2,4,figsize=(12,6))
for row,p in enumerate([target,chosen]):
    c=pd.read_csv(p/'camera.csv')
    for ax,i in zip(axs[row],[0,len(c)//2,len(c)-15,len(c)-1]):
        k=int(c.frame.iloc[i]);img=cv2.imdecode(np.fromfile(p/'frames'/f'frame_{k:06d}.jpg',dtype=np.uint8),cv2.IMREAD_COLOR)
        ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));ax.set_title(p.name.rsplit('_',1)[1]+' '+str(k));ax.axis('off')
fig.tight_layout();fig.savefig(base/'replacement_frames_review.png',dpi=120)
r.to_csv(base/'replacement_candidate_ranking.csv',index=False)
