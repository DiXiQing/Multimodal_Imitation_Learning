from pathlib import Path
import argparse
import hashlib
import shutil
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'Data/Exprement Data/Khairy/3'
REVIEW=ROOT/'Report/khairy_selected_15'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true');args=parser.parse_args()
    REVIEW.mkdir(parents=True,exist_ok=True)
    records=[];inputs={}
    for trial in sorted(BASE.glob('TRIAL_*')):
        c=pd.read_csv(trial/'camera.csv');d=pd.read_csv(trial/'grasp_data_smoothed.csv')
        m=pd.read_csv(trial/'imu.csv').sort_values('t_imu').drop_duplicates('t_imu',keep='last')
        y=c.finger_width.to_numpy();a=c.object_area.to_numpy();peak=int(np.argmax(d.finger_width))
        start=min(10,peak);baseline=np.median(a[:start]);ap=a[:peak+1]
        records.append(dict(trial=trial.name,rows=len(c),initial_fw=np.median(y[:5]),peak_fw=d.finger_width.max(),
            peak_progress=peak/(len(c)-1)*100,oa_min_ratio=ap.min()/baseline,
            oa_end_ratio=np.median(a[-10:])/baseline,fw_tail_range=np.ptp(y[-12:]),
            oa_tail_relative_range=np.ptp(a[-12:])/np.median(a[-12:]),
            initial_abs_acc=np.median(np.abs(d.forward_raw.iloc[:5])),imu_max_gap=np.diff(m.t_imu).max(),
            camera_max_gap=np.diff(c.t_camera).max()))
        inputs[trial.name]=(trial,c,d,peak)
    quality=pd.DataFrame(records);quality.to_csv(REVIEW/'all_candidates.csv',index=False)
    print(quality.to_string(index=False),flush=True)
    for page in range(0,len(quality),6):
        rows=quality.iloc[page:page+6];fig,axs=plt.subplots(len(rows),5,figsize=(16,2.5*len(rows)),squeeze=False)
        for row,r in enumerate(rows.itertuples()):
            trial,c,d,peak=inputs[r.trial]
            points=[0,max(0,peak-10),peak,min(len(c)-1,peak+8),len(c)-1]
            for ax,k in zip(axs[row],points):
                f=int(c.frame.iloc[k]);im=cv2.imdecode(np.fromfile(trial/'frames'/f'frame_{f:06d}.jpg',np.uint8),1)
                ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));ax.axis('off')
                ax.set_title(f'{r.trial[-6:]} frame {f} FW={c.finger_width.iloc[k]:.0f}',fontsize=9)
        fig.tight_layout();fig.savefig(REVIEW/f'candidate_frames_{page//6+1}.jpg',dpi=110);plt.close(fig)
    if not args.export:return
    chosen=pd.read_csv(REVIEW/'selection_manifest.csv')
    names=chosen.loc[chosen.selected,'trial'].tolist()
    assert len(names)==15 and len(set(names))==15
    destination=BASE.parent/'selected_15_3cm'
    if destination.exists():raise ValueError(f'Output already exists: {destination}')
    destination.mkdir()
    for name in names:
        source=inputs[name][0];target=destination/name
        shutil.copytree(source,target)
        for path in source.rglob('*'):
            if path.is_file():assert hashlib.sha256(path.read_bytes()).digest()==hashlib.sha256((target/path.relative_to(source)).read_bytes()).digest()
        c=pd.read_csv(target/'camera.csv');raw=pd.read_csv(target/'grasp_data.csv');smoothed=pd.read_csv(target/'grasp_data_smoothed.csv')
        assert np.array_equal(c.frame,raw.frame) and np.array_equal(c.frame,smoothed.frame)
        assert np.allclose(c.t_camera,raw.t_camera,atol=1e-6,rtol=0) and np.allclose(c.t_camera,smoothed.t_camera,atol=1e-6,rtol=0)
        assert np.array_equal(raw.forward_raw,smoothed.forward_raw)
        assert len(list((target/'frames').glob('*.jpg')))==len(c)
        cap=cv2.VideoCapture(str(target/'grasp_video.mp4'));video_count=0
        while True:
            ok,_=cap.read()
            if not ok:break
            video_count+=1
        cap.release();assert video_count==len(c)
        m=pd.read_csv(target/'imu.csv').sort_values('t_imu').drop_duplicates('t_imu',keep='last')
        assert m.t_imu.iloc[0]<=c.t_camera.iloc[0] and m.t_imu.iloc[-1]>=c.t_camera.iloc[-1]
        assert np.diff(m.t_imu).max()<.25
        print('Copied',name,flush=True)
    chosen.to_csv(destination/'selection_manifest.csv',index=False,encoding='utf-8-sig')
    rng=np.random.default_rng(42)
    test=set(rng.choice(sorted(names),size=3,replace=False))
    split=pd.DataFrame([dict(trial=n,object_size_cm=3.,split='test' if n in test else 'train') for n in sorted(names)])
    split.to_csv(destination/'data_split.csv',index=False)
    fig,axs=plt.subplots(3,1,figsize=(13,10),sharex=True)
    for i,name in enumerate(names):
        d=inputs[name][2];x=100*(d.frame-d.frame.iloc[0])/(d.frame.iloc[-1]-d.frame.iloc[0])
        for ax,col in zip(axs,['finger_width','object_area','forward_acceleration']):ax.plot(x,d[col],lw=1,color=plt.cm.turbo(i/14),alpha=.7,label=name[-6:])
    for ax,label in zip(axs,['FW (px)','OA (px²)','Signed forward acceleration (m/s²)']):ax.set_ylabel(label);ax.grid(alpha=.2)
    axs[-1].set_xlabel('Segment progress (%)');axs[-1].axhline(0,color='black',lw=.5)
    fig.suptitle('15 trials selected by acquisition quality and raw frames; no prediction-error selection')
    fig.tight_layout();fig.savefig(destination/'selected_overlay.png',dpi=150);plt.close(fig)
    print('Output',destination,flush=True)


if __name__=='__main__':main()
