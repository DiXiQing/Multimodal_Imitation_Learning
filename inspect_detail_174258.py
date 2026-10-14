exec(open('inspect_recording_174258.py',encoding='utf-8-sig').read().split('for page in range')[0].split('fig,axs=')[0])
from datetime import datetime,timezone,timedelta
fig,axs=plt.subplots(5,5,figsize=(15,12))
for ax,k in zip(axs.flat,range(6550,7050,20)):
 im=cv2.imdecode(np.fromfile(source/'frames'/f'frame_{k:06d}.jpg',np.uint8),1)
 ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));ax.set_title(f'{k} '+datetime.fromtimestamp(c.t_camera.iloc[k],timezone(timedelta(hours=9))).strftime('%H:%M:%S'));ax.axis('off')
fig.tight_layout();fig.savefig(out/'size_transition.jpg',dpi=130);plt.close(fig)
fig,axs=plt.subplots(5,1,figsize=(18,13))
for ax,(a,b) in zip(axs,[(60,330),(1800,1990),(2660,2810),(6960,7130),(11380,11617)]):
 ax.plot(c.frame.iloc[a:b],y[a:b]);ax.set_ylim(0,240);ax.grid();ax2=ax.twinx();ax2.plot(c.frame.iloc[a:b],c.object_area.iloc[a:b],color='orange');ax2.set_ylim(0,12000)
fig.tight_layout();fig.savefig(out/'detail_signals.png',dpi=110)
m=m.sort_values('t_imu').drop_duplicates('t_imu',keep='last');dt=m.t_imu.diff();g=m.loc[dt>.08,['imu_index','t_imu']].copy();g['gap']=dt[dt>.08];g['frame']=np.interp(g.t_imu,c.t_camera,c.frame);print(g.to_string(index=False));g.to_csv(out/'imu_gaps.csv',index=False)
