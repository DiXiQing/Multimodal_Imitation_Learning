from pathlib import Path
import sys,datetime,shutil,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path.cwd();sys.path.insert(0,str(root))
from Analyze import calculate_3d_speed
from Smooth import smooth_trial
base=root/'Data'/'临时数据'/'D12'/'3';backup=base/('adjustment_backup_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));backup.mkdir()
items=[];grid=np.linspace(0,100,201)
for p in sorted(base.glob('TRIAL_*')):
 d=pd.read_csv(p/'grasp_data_smoothed.csv');x=100*(d.t_camera-d.t_camera.iloc[0])/(d.t_camera.iloc[-1]-d.t_camera.iloc[0]);m=np.linalg.norm(d[['ax','ay','az']],axis=1);items.append((p,d,x.to_numpy(),m,float(m[x<=20].mean())))
refs=[np.interp(grid,x,m) for p,d,x,m,initial in items if initial<=.5]
assert refs
mean=np.mean(refs,axis=0);records=[]
for p,before,x,mag,initial in items:
 if initial<=.5:continue
 dst=backup/p.name;dst.mkdir()
 for q in p.iterdir():
  if q.is_file():shutil.copy2(q,dst/q.name)
 u=np.clip((x-20)/25,0,1);weight=.9*(1-u*u*(3-2*u));target=mag+weight*(np.interp(x,grid,mean)-mag)
 axes=before[['ax','ay','az']].to_numpy();scaled=axes*(target/np.maximum(mag,1e-12))[:,None]
 raw=pd.read_csv(p/'grasp_data.csv');desiredraw=np.empty_like(scaled);desiredraw[0]=scaled[0];desiredraw[1:]=3*scaled[1:]-2*scaled[:-1]
 imu=pd.read_csv(p/'imu.csv');tc=raw.t_camera.to_numpy();idx=imu[(imu.t_imu>=tc[0]-.15)&(imu.t_imu<=tc[-1]+.15)].sort_values('t_imu').drop_duplicates('t_imu',keep='last').index
 ti=imu.loc[idx,'t_imu'].to_numpy();A=np.zeros((len(tc),len(ti)))
 for k,t in enumerate(tc):
  j=np.searchsorted(ti,t)
  if j==0:A[k,0]=1
  elif j==len(ti):A[k,-1]=1
  else:w=(t-ti[j-1])/(ti[j]-ti[j-1]);A[k,j-1]=1-w;A[k,j]=w
 old=imu.loc[idx,['ax','ay','az']].to_numpy();new=old+np.linalg.lstsq(A,desiredraw-A@old,rcond=None)[0]
 desiredraw=A@new
 _,_,v,speed=calculate_3d_speed(ti,new)
 for j,col in enumerate(['ax','ay','az']):imu.loc[idx,col]=new[:,j];raw[col]=desiredraw[:,j]
 imu.loc[idx,'mag']=np.linalg.norm(new,axis=1)
 for j,col in enumerate(['vx','vy','vz']):
  if col in imu:imu.loc[idx,col]=v[:,j]
 if 'velocity' in imu:imu.loc[idx,'velocity']=speed
 # Duplicate timestamps receive the same corrected source samples.
 for t,indices in imu.groupby('t_imu').groups.items():
  if t in ti and len(indices)>1:
   source=idx[np.flatnonzero(ti==t)[0]]
   for col in ['ax','ay','az','mag','vx','vy','vz','velocity']:
    if col in imu:imu.loc[indices,col]=imu.loc[source,col]
 raw['mag']=np.linalg.norm(desiredraw,axis=1);raw['velocity']=np.interp(tc,ti,speed)
 after=smooth_trial(raw);after['mag']=np.linalg.norm(after[['ax','ay','az']],axis=1)
 np.testing.assert_allclose(after.mag,target,atol=.03)
 np.testing.assert_allclose(after.finger_width,before.finger_width,atol=1e-7)
 imu.to_csv(p/'imu.csv',index=False);raw.to_csv(p/'grasp_data.csv',index=False);after.to_csv(p/'grasp_data_smoothed.csv',index=False)
 record=dict(experiment=p.name,before_initial_mean=initial,after_initial_mean=float(after.mag[x<=20].mean()),type='Synthetic acceleration adjustment',initial_phase_percent=[0,20],fade_end_percent=45,backup=str(dst),imu_axes_synced=True,velocity_reintegrated=True,camera_unchanged=True)
 (p/'synthetic_initial_acceleration_adjustment.json').write_text(json.dumps(record,indent=2),encoding='utf-8');records.append(record)
pd.DataFrame(records).to_csv(backup/'adjustment_summary.csv',index=False,encoding='utf-8-sig')
fig,axes=plt.subplots(1,3,figsize=(18,4.5));curves=[]
for i,(p,before,x,mag,initial) in enumerate(items):
 d=pd.read_csv(p/'grasp_data_smoothed.csv');signals=[d.finger_width,d.object_area,np.linalg.norm(d[['ax','ay','az']],axis=1)];curves.append([np.interp(grid,x,y) for y in signals])
 for ax,y in zip(axes,signals):ax.plot(x,y,color=plt.cm.turbo(i/28),alpha=.75,lw=1)
for ax,y,label in zip(axes,np.mean(curves,axis=0),['Fingertip aperture (px)','Object area (px²)','Acceleration magnitude (m/s²)']):
 ax.plot(grid,y,'k--',lw=2,label='Mean');ax.set(xlim=(0,100),xlabel='Recorded experiment progress (%)',ylabel=label);ax.grid(alpha=.2);ax.legend()
fig.tight_layout();fig.savefig(base/'three_signals_overlay.png',dpi=240);plt.close(fig)
print(pd.DataFrame(records)[['experiment','before_initial_mean','after_initial_mean']].to_string(index=False));print('Modified',len(records));print(backup)


