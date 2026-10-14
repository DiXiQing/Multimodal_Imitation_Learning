exec(open('crop_recording_174258.py',encoding='utf-8-sig').read().split('records=[]')[0])
fig,axs=plt.subplots(3,1,figsize=(15,9),sharex=True)
sel=c.iloc[11190:11360]
axs[0].plot(sel.frame,sel.finger_width);axs[1].plot(sel.frame,sel.object_area)
mi=imu[imu.t_imu.between(sel.t_camera.iloc[0],sel.t_camera.iloc[-1])];axs[2].plot(np.interp(mi.t_imu,c.t_camera,c.frame),mi.forward_raw)
for ax in axs:
 ax.grid();ax.set_xticks(np.arange(11190,11361,10))
fig.tight_layout();fig.savefig(out/'short_rest_detail.png',dpi=100)
