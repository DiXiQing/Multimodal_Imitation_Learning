from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,shutil,hashlib,tempfile
import numpy as np
import pandas as pd
import cv2
from scipy.signal import find_peaks,butter,sosfilt,sosfilt_zi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from Smooth import smooth_trial


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path)
    parser.add_argument('--export',action='store_true')
    args=parser.parse_args()
    source=args.source.resolve()
    camera=pd.read_csv(source/'camera.csv').sort_values('frame').reset_index(drop=True)
    imu=pd.read_csv(source/'imu.csv').sort_values('t_imu').drop_duplicates('t_imu',keep='last')
    y=camera.finger_width.rolling(5,center=True,min_periods=1).median().to_numpy()
    peaks,_=find_peaks(y,prominence=80,distance=70)
    records=[]
    for peak in peaks:
        if y[peak]<150 or peak<40:continue
        look=max(0,peak-100);low=float(np.min(y[look:peak]))
        below=np.flatnonzero(y[look:peak]<low+.1*(y[peak]-low))
        if not len(below):continue
        onset=look+int(below[-1])+1;start=max(0,onset-25)
        stable=next((j for j in range(peak+8,min(len(y)-9,peak+80)) if low+.1*(y[peak]-low)<=np.median(y[j:j+9])<=low+.7*(y[peak]-low) and np.ptp(y[j:j+9])<=10),None)
        if stable is None:continue
        end=min(len(y)-1,stable+15)
        gain=float(camera.object_area.iloc[stable:end+1].median()/max(camera.object_area.iloc[start:onset+1].median(),1))
        if gain<1.25:continue
        records.append(dict(start=int(start),end=int(end),peak=int(peak),stable=int(stable),area_gain=gain))
    pd.DataFrame(records).to_csv(source/'crop_ranges.csv',index=False)
    for page in range(0,len(records),6):
        chosen=records[page:page+6];fig,axs=plt.subplots(len(chosen),4,figsize=(12,2.4*len(chosen)),squeeze=False)
        for row,r in enumerate(chosen):
            for ax,key in zip(axs[row],['start','peak','stable','end']):
                k=int(camera.frame.iloc[r[key]]);im=cv2.imdecode(np.fromfile(source/'frames'/f'frame_{k:06d}.jpg',dtype=np.uint8),cv2.IMREAD_COLOR)
                ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));ax.set_title(f'{page+row+1}: {key} {k}');ax.axis('off')
        fig.tight_layout();fig.savefig(source/f'crop_frames_{page//6+1}.png',dpi=110);plt.close(fig)
    print(pd.DataFrame(records).to_string(index=False),flush=True)
    if not args.export:return
    output=source/'grasp_segments';output.mkdir(exist_ok=True)
    originals={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in ['camera.csv','imu.csv','recorded_video.mp4']}
    sos=butter(2,3,fs=50,output='sos');curves=[];summary=[]
    for number,r in enumerate(records,1):
        d=camera.iloc[r['start']:r['end']+1].copy();tc=d.t_camera.to_numpy()
        before=imu.loc[imu.t_imu<=tc[0]].tail(1);after=imu.loc[imu.t_imu>=tc[-1]].head(1)
        if before.empty or after.empty:print('Skip: IMU coverage',number);continue
        seg=imu.loc[imu.t_imu.between(before.t_imu.iloc[0],after.t_imu.iloc[0])];ti=seg.t_imu.to_numpy();gap=float(np.diff(ti).max())
        if gap>.25:print('Skip: IMU gap',number,gap);continue
        stamp=datetime.fromtimestamp(float(tc[0]),timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')
        size=source.name.split('Size_')[1].split('cm_')[0];dest=output/f'TRIAL_Size_{size}cm_{stamp}'
        if dest.exists():raise RuntimeError(f'Already exists: {dest}')
        axes=seg[['ax','ay','az']].to_numpy()
        for j,col in enumerate(['ax','ay','az']):d[col]=np.interp(tc,ti,axes[:,j])
        d['mag']=np.linalg.norm(d[['ax','ay','az']],axis=1)
        if 'velocity' in seg:d['velocity']=np.interp(tc,ti,seg.velocity)
        if 'forward_raw' in seg:d['forward_raw']=np.interp(tc,ti,seg.forward_raw)
        uniform=ti[0]+np.arange(int(np.floor((ti[-1]-ti[0])*50))+1)/50
        u=np.column_stack([np.interp(uniform,ti,axes[:,j]) for j in range(3)])
        filtered=np.column_stack([sosfilt(sos,u[:,j],zi=sosfilt_zi(sos)*u[0,j])[0] for j in range(3)])
        sync=np.column_stack([np.interp(tc,uniform,filtered[:,j]) for j in range(3)])
        smooth=smooth_trial(d,sync)
        if 'forward_raw' in seg:
            forward=np.interp(uniform,ti,seg.forward_raw);sf=sosfilt(sos,forward,zi=sosfilt_zi(sos)*forward[0])[0]
            smooth['forward_acceleration']=np.interp(tc,uniform,sf)
        dest.mkdir();frames=dest/'frames';frames.mkdir()
        d.to_csv(dest/'grasp_data.csv',index=False);smooth.to_csv(dest/'grasp_data_smoothed.csv',index=False)
        camera.iloc[r['start']:r['end']+1].to_csv(dest/'camera.csv',index=False);seg.to_csv(dest/'imu.csv',index=False)
        fd=tempfile.NamedTemporaryFile(suffix='.mp4',delete=False);temp=Path(fd.name);fd.close();writer=None
        try:
            for frame in d.frame.astype(int):
                fp=source/'frames'/f'frame_{frame:06d}.jpg';im=cv2.imdecode(np.fromfile(fp,dtype=np.uint8),cv2.IMREAD_COLOR)
                if writer is None:
                    h,w=im.shape[:2];writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),30,(w,h))
                    if not writer.isOpened():raise RuntimeError('Video writer failed')
                writer.write(im);shutil.copy2(fp,frames/fp.name)
        finally:
            if writer is not None:writer.release()
        cap=cv2.VideoCapture(str(temp));count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release();assert count==len(d)
        shutil.move(str(temp),str(dest/'grasp_video.mp4'))
        summary.append(dict(trial=dest.name,frame_start=int(d.frame.iloc[0]),frame_end=int(d.frame.iloc[-1]),rows=len(d),imu_max_gap_seconds=gap,fw_zero=int((d.finger_width<=0).sum()),oa_zero=int((d.object_area<=0).sum())))
        curves.append((dest.name,smooth));print('Saved',dest.name,len(d),flush=True)
    fig,axs=plt.subplots(2,2,figsize=(16,9))
    for i,(name,d) in enumerate(curves):
        x=100*(d.frame-d.frame.iloc[0])/(d.frame.iloc[-1]-d.frame.iloc[0])
        for ax,col in zip(axs.flat,['finger_width','object_area','mag','forward_acceleration']):
            if col in d:ax.plot(x,d[col],color=plt.cm.turbo(i/max(1,len(curves)-1)),lw=1.2,label=name.rsplit('_',1)[1])
    for ax,label in zip(axs.flat,['FW (px)','OA (px2)','Acceleration magnitude (m/s2)','Signed forward acceleration (m/s2)']):
        ax.set(xlabel='Trial progress (%)',ylabel=label,xlim=(0,100));ax.grid(alpha=.2)
    axs[1,1].axhline(0,color='black',lw=.7)
    h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=6,fontsize=8,title=f'{len(curves)} experiments')
    fig.tight_layout(rect=(0,.13,1,1));fig.savefig(output/'combined_overlay.png',dpi=150);plt.close(fig)
    pd.DataFrame(summary).to_csv(output/'crop_summary.csv',index=False)
    assert originals=={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in originals}
    print('Completed',len(curves),'source files unchanged')


if __name__=='__main__':main()
