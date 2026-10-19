"""Reviewed segmentation of this continuous recording; no model fitting."""
import csv
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'Data/TopView/Inference/run_20261018_153832_802120/trial_001'
# Six complete grasps checked against raw video. Exchanges and incomplete reach near 29.98 s excluded.
PEAK_TIMES = [3.2982616,12.8356434,22.7240488,33.0287879,41.5236621,49.3311248]
EXPECTED_COUNT = 6
EXCLUDED_EXCHANGE_TIMES = [10.0,28.8,29.98]


def main():
    with (SOURCE/'predictions.csv').open(encoding='utf-8-sig') as f:
        reader = csv.DictReader(f);fields = list(reader.fieldnames);rows = list(reader)
    t = np.array([float(r['trial_elapsed_s']) for r in rows])
    def values(key):return np.array([float(r[key]) if r[key] else np.nan for r in rows])
    b,d,pred = values('measured_aperture_px'),values('distance_px'),values('predicted_aperture_px')
    out = SOURCE/'grasp_segments'
    out.mkdir(exist_ok=False)
    plans = []
    for number,pt in enumerate(PEAK_TIMES,1):
        peak = int(np.argmin(abs(t-pt)));onset = peak
        while onset>0 and b[onset]>20 and t[peak]-t[onset]<3:onset-=1
        start = max(0,onset-12)
        after = np.arange(peak+1,min(len(rows),peak+100))
        near = after[np.isfinite(d[after])&(d[after]<25)]
        hold = float(np.nanmedian(b[near[:55]]))
        stable = None
        for j in after:
            if j+10<len(rows) and np.isfinite(b[j:j+10]).all() and np.max(abs(b[j:j+10]-hold))<5 and np.nanmedian(d[j:j+10])<25:
                stable=int(j);break
        if stable is None:raise ValueError(f'No stable hold: {pt}')
        end = min(len(rows)-1,stable+20)
        candidates = [r['selected_model_color'] for r in rows[max(0,peak-15):peak+16] if r['selected_model_color']]
        color = max(set(candidates),key=candidates.count);size={'red':2,'blue':3,'black':4}[color]
        stamp = datetime.fromtimestamp(float(rows[start]['unix_time_s']),timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')
        name=f'{color.title()}_{size}cm_{stamp}';folder=out/name;folder.mkdir()
        segment=dict(number=number,folder=name,color=color,size_cm=size,start=start,onset=onset,peak=peak,
            stable=stable,end=end,frames=end-start+1,start_time_s=float(t[start]),end_time_s=float(t[end]),
            source=str(SOURCE),boundary_rule='Reviewed grasp peaks; 12 frames before onset, 20 frames after stable hold begins; withdrawal/object exchange and incomplete reach excluded.')
        segment_rows=[]
        for i in range(start,end+1):
            segment_rows.append(dict(segment_frame=i-start,segment_elapsed_s=float(t[i]-t[start]),**rows[i]))
        with (folder/'predictions.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=['segment_frame','segment_elapsed_s',*fields]);w.writeheader();w.writerows(segment_rows)
        (folder/'segment.json').write_text(json.dumps(segment,indent=2),encoding='utf-8')
        error=pred[start:end+1]-b[start:end+1];valid=np.isfinite(error)
        summary=dict(total_frames=end-start+1,predicted_frames=int(valid.sum()),
            mae_px=float(np.mean(abs(error[valid]))) if valid.any() else None,
            rmse_px=float(np.sqrt(np.mean(error[valid]**2))) if valid.any() else None,
            note='Original recorded predictions unchanged, not recomputed. Missing predictions remain blank.')
        (folder/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        fig,ax=plt.subplots(figsize=(10,4));sl=slice(start,end+1)
        ax.plot(t[sl]-t[start],b[sl],label='Measured',color='#222222')
        ax.plot(t[sl]-t[start],pred[sl],label='Predicted',color='#e55236')
        ax.set(title=name,xlabel='Segment time (s)',ylabel='Aperture (px)');ax.grid(alpha=.2);ax.legend()
        fig.tight_layout();fig.savefig(folder/'prediction_plot.png',dpi=160);plt.close(fig)
        plans.append(segment)
    # Original frame indices are authoritative: videos were saved at fixed FPS.
    for filename in ['raw_video.mp4','prediction_video.mp4']:
        cap=cv2.VideoCapture(str(SOURCE/filename))
        if not cap.isOpened():raise RuntimeError(filename)
        fps=cap.get(cv2.CAP_PROP_FPS);size=(int(cap.get(3)),int(cap.get(4)))
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==len(rows)
        for segment in plans:
            cap.set(cv2.CAP_PROP_POS_FRAMES,segment['start'])
            writer=cv2.VideoWriter(str(out/segment['folder']/filename),cv2.VideoWriter_fourcc(*'mp4v'),fps,size)
            if not writer.isOpened():raise RuntimeError('Video writer failed')
            try:
                for i in range(segment['frames']):
                    ok,frame=cap.read()
                    if not ok:raise RuntimeError('Video ended before planned boundary')
                    writer.write(frame)
            finally:writer.release()
            check=cv2.VideoCapture(str(out/segment['folder']/filename))
            assert int(check.get(cv2.CAP_PROP_FRAME_COUNT))==segment['frames'];check.release()
        cap.release()
    colors=[c for c in ['red','blue','black'] if any(s['color']==c for s in plans)]
    fig,axes=plt.subplots(1,len(colors),figsize=(5.5*len(colors),5),sharex=True,sharey=True)
    for ax,color in zip(np.atleast_1d(axes),colors):
        subset=[s for s in plans if s['color']==color]
        for s in subset:
            sl=slice(s['start'],s['end']+1)
            ax.plot(d[sl],b[sl],label=s['folder'][-6:],lw=1.4)
        ax.set(title=f'{color.title()} | {len(subset)} grasps',xlabel='Hand-object distance (px)',ylabel='Measured aperture (px)')
        ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(out/'distance_aperture_overlay.png',dpi=170);plt.close(fig)
    columns=min(4,len(plans));nrows=int(np.ceil(len(plans)/columns))
    fig,axes=plt.subplots(nrows,columns,figsize=(4.25*columns,3.5*nrows),squeeze=False)
    for ax in list(axes.flat)[len(plans):]:ax.axis('off')
    for ax,s in zip(axes.flat,plans):
        sl=slice(s['start'],s['end']+1);ax.plot(t[sl]-t[s['start']],b[sl],color='#222222',label='Measured')
        ax.plot(t[sl]-t[s['start']],pred[sl],color='#e55236',label='Predicted')
        ax.set(title=f'{s["number"]:02d} {s["color"]}',xlabel='Time (s)',ylabel='Aperture (px)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'prediction_overview.png',dpi=160);plt.close(fig)
    counts={c:sum(s['color']==c for s in plans) for c in ['red','blue','black']}
    (out/'segments_manifest.json').write_text(json.dumps(dict(expected_by_user=EXPECTED_COUNT,confirmed_grasps=len(plans),
        counts=counts,excluded_exchange_times_s=EXCLUDED_EXCHANGE_TIMES,segments=plans),indent=2),encoding='utf-8')
    print(f'Saved {len(plans)} reviewed grasps: {counts}. {out}')


if __name__=='__main__':main()
