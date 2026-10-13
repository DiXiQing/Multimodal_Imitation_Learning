from pathlib import Path
import sys,json,hashlib,shutil,datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from Analyze import calculate_3d_speed
from Smooth import smooth_trial
base=root/'Data'/'临时数据'/'D13';group=base/'01_stable_calibration';review=base/'review'
trials=sorted(group.glob('TRIAL_*'));assert len(trials)==15
backup=root/'analysis'/('D13_before_start_extension_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));backup.mkdir()
curves={};records=[]
for t in trials:
    old=pd.read_csv(t/'grasp_data.csv');meta=json.loads((t/'crop_metadata.json').read_text())
    oldstart=int(meta['frame_start']);end=int(meta['frame_end']);start=oldstart-10
    c=pd.read_csv(t/'camera.csv');im=pd.read_csv(t/'imu.csv').sort_values('t_imu').drop_duplicates('t_imu',keep='last')
    d=c.loc[c.frame.between(start,end)].copy()
    assert start>=0 and len(d)==int(meta['rows'])+10 and int(d.frame.iloc[0])==start and int(d.frame.iloc[-1])==end
    source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [t/'camera.csv',t/'imu.csv',t/'recorded_video.mp4']}
    b=backup/t.name;b.mkdir()
    for name in ['grasp_data.csv','grasp_data_smoothed.csv','crop_metadata.json','smoothing_preview.png','grasp_video.mp4']:
        p=t/name
        if p.exists():shutil.copy2(p,b/name)
    tc=d.t_camera.to_numpy();segment=im.loc[im.t_imu.between(tc[0],tc[-1])];ti=segment.t_imu.to_numpy()
    assert len(ti)>10 and np.diff(ti).max()<.25
    a=segment[['ax','ay','az']].to_numpy();_,_,_,speed=calculate_3d_speed(ti,a)
    for j,col in enumerate(['ax','ay','az']):d[col]=np.interp(tc,ti,a[:,j])
    d['mag']=np.linalg.norm(d[['ax','ay','az']],axis=1);d['velocity']=np.interp(tc,ti,speed)
    smooth=smooth_trial(d);smooth['mag']=np.linalg.norm(smooth[['ax','ay','az']],axis=1)
    d.to_csv(t/'grasp_data.csv',index=False);smooth.to_csv(t/'grasp_data_smoothed.csv',index=False,float_format='%.9g')
    fig,axs=plt.subplots(3,1,figsize=(11,8),sharex=True)
    for ax,col,label in zip(axs,['finger_width','object_area','mag'],['Finger width (px)','Object area (px²)','Acceleration (m/s²)']):
        ax.plot(d.frame,d[col],alpha=.4,label='Raw');ax.plot(smooth.frame,smooth[col],label='Smoothed');ax.axvline(oldstart,color='gray',ls=':',label='Previous start');ax.set_ylabel(label);ax.grid(alpha=.2)
    axs[0].legend();axs[-1].set_xlabel('Frame');fig.tight_layout();fig.savefig(t/'smoothing_preview.png',dpi=140);plt.close(fig)
    images=[t/'frames'/f'frame_{int(n):06d}.jpg' for n in d.frame]
    assert all(p.exists() for p in images)
    first=cv2.imdecode(np.fromfile(images[0],dtype=np.uint8),cv2.IMREAD_COLOR);assert first is not None
    h,w=first.shape[:2];clip=t/'grasp_video.mp4'
    temporary_clip=root/'analysis'/f'd13_clip_{t.name[-6:]}.mp4'
    writer=cv2.VideoWriter(str(temporary_clip),cv2.VideoWriter_fourcc(*'mp4v'),30.,(w,h));assert writer.isOpened()
    try:
        for p in images:
            frame=cv2.imdecode(np.fromfile(p,dtype=np.uint8),cv2.IMREAD_COLOR);assert frame is not None and frame.shape[:2]==(h,w)
            writer.write(frame)
    finally:writer.release()
    cap=cv2.VideoCapture(str(temporary_clip));assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==len(d);cap.release()
    shutil.move(str(temporary_clip),str(clip))
    meta.update(frame_start=start,frame_end=end,rows=len(d),previous_frame_start=oldstart,
                initial_acceleration=float(smooth.mag.iloc[:5].mean()),final_acceleration=float(smooth.mag.iloc[-5:].mean()),
                acc_peak=float(smooth.mag.max()),fw_peak=float(smooth.finger_width.max()),start_extension_frames=10,
                extension_backup=str(b),clip_file='grasp_video.mp4',clip_fps=30,
                method='Original crop extended with 10 preceding recorded frames; end unchanged; source CSV/video unchanged; speed and causal smoothing recalculated')
    (t/'crop_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    assert source=={name:hashlib.sha256((t/name).read_bytes()).hexdigest() for name in source}
    records.append({k:v for k,v in meta.items() if k not in ['source_sha256','method']});curves[t.name]=smooth
    print(t.name[-6:],oldstart,'->',start,'end',end,'rows',len(d),flush=True)
pd.DataFrame(records).to_csv(review/'crop_and_group_summary.csv',index=False,encoding='utf-8-sig')
(review/'grouping_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
fig,axs=plt.subplots(2,2,figsize=(16,8));signal_axes=[axs[0,0],axs[0,1],axs[1,0]]
for i,(name,d) in enumerate(curves.items()):
    x=100*(d.frame-d.frame.iloc[0])/(d.frame.iloc[-1]-d.frame.iloc[0])
    for ax,col in zip(signal_axes,['finger_width','object_area','mag']):ax.plot(x,d[col],color=plt.cm.tab20(i/max(1,len(trials)-1)),lw=1.2,label=name[-6:])
for ax,label in zip(signal_axes,['Finger width (px)','Object area (px²)','Acceleration magnitude (m/s²)']):ax.set(xlabel='Recorded trial progress (%)',ylabel=label,xlim=(0,100));ax.grid(alpha=.2)
axs[1,1].axis('off');h,l=signal_axes[0].get_legend_handles_labels();axs[1,1].legend(h,l,loc='center',ncol=3,frameon=False,title=f'{len(trials)} trials — start extended by 10 frames');fig.tight_layout();fig.savefig(review/'01_stable_calibration_overlay.png',dpi=150);plt.close(fig)
print('Backup:',backup)
