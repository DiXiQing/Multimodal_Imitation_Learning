"""Auditable segmentation for the black 3 cm recording; raw measurements unchanged."""
import os
from pathlib import Path
os.environ['MPLCONFIGDIR'] = str(Path(__file__).parent / '.mplconfig')
import csv, json, hashlib, shutil
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SOURCE = Path(__file__).resolve().parents[1]/'Data/TopView/TRIAL_Color_black_Size_3cm_20261029_145748_073681'
OUT = SOURCE/'grasp_segments'

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def main():
    OUT.mkdir(exist_ok=True)
    originals={p.name:digest(p) for p in SOURCE.iterdir() if p.is_file()}
    settings=json.loads((SOURCE/'settings.json').read_text(encoding='utf-8-sig'))
    with (SOURCE/'camera.csv').open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    def values(col):return np.array([float(r[col]) if r.get(col) else np.nan for r in rows])
    d=values('index_mcp_object_distance');a=values('finger_width');t=values('t_camera')
    scale=settings.get('pixel_calibration',{}).get('mm_per_px')
    calibrated=scale is not None and float(scale)>0
    unit='mm' if calibrated else 'px'
    dc='index_mcp_object_distance_mm' if calibrated else 'index_mcp_object_distance'
    ac='finger_width_mm' if calibrated else 'finger_width'
    dd=values(dc);aa=values(ac)
    if calibrated:
        assert np.allclose(dd[np.isfinite(d)],d[np.isfinite(d)]*scale)
        assert np.allclose(aa[np.isfinite(a)],a[np.isfinite(a)]*scale)
    xy=np.hypot(values('index_mcp_x')-values('object_x'),values('index_mcp_y')-values('object_y'))
    assert np.allclose(d[np.isfinite(d)],xy[np.isfinite(d)])
    def runs(mask):
        e=np.diff(np.r_[False,mask,False].astype(int))
        return list(zip(np.flatnonzero(e==1).tolist(),(np.flatnonzero(e==-1)-1).tolist()))
    near=[]
    for s,e in runs(d<55):
        if near and s-near[-1][1]-1<=8:near[-1]=(near[-1][0],e)
        else:near.append((s,e))
    entries=[];reviews=[];previous=0
    for n,(s,e) in enumerate(near,1):
        lo=previous;previous=e+1
        peak=lo+int(np.nanargmax(a[lo:s+1]))
        # Last closed, distant rest before opening; adaptive local baseline.
        rest=[i for i in range(lo+10,peak) if np.all(a[i-10:i+1]<15) and np.all(d[i-10:i+1]>230)]
        if rest:
            anchor=rest[-1]
            baseline_d=float(np.median(d[anchor-10:anchor+1]));baseline_a=float(np.median(a[anchor-10:anchor+1]))
            onset=anchor
            # Backtrack from the 15 px opening gate to first sustained departure.
            while onset>lo+10 and (np.max(a[onset-2:onset+1])>baseline_a+3 or np.min(d[onset-2:onset+1])<baseline_d-4):onset-=1
        else:onset=max(lo,peak-30)
        # New session has slow pre-opening approach: backtrack to quiet pre-roll.
        while onset>lo+10 and (np.ptp(d[onset-10:onset])>4 or np.ptp(a[onset-10:onset])>3):
            onset-=1
        start=max(lo,onset-10)
        target=float(np.nanpercentile(a[max(s,peak+1):e+1],25))
        candidates=[i for i in range(max(s,peak+1),e-13) if np.all(np.isfinite(a[i:i+15])) and np.max(a[i:i+5])<=target+2.5 and np.ptp(a[i:i+5])<=3 and np.all(d[i:i+15]<46) and np.max(a[i:i+15])<=target+5]
        flags=[]
        if not rest:flags.append('No confirmed distant closed rest before action')
        if candidates:stable=candidates[0]+4;end=stable+10
        else:
            stable=None;end=e;flags.append('No confirmed closure plus 10-frame hold; retained for review')
        if onset-start!=10:flags.append('Insufficient pre-action margin')
        if np.ptp(d[start:onset])>4 or np.ptp(a[start:onset])>3:flags.append('Pre-action stillness uncertain')
        if np.any(~np.isfinite(d[start:end+1])) or np.any(~np.isfinite(a[start:end+1])):flags.append('Missing measurements')
        if np.any(np.diff(t[start:end+1])<=0):flags.append('Non-increasing timestamps')
        invalid=sum(r.get('valid')!='1' or r.get('index_mcp_distance_valid')!='1' for r in rows[start:end+1])
        if invalid:flags.append(f'{invalid} invalid detections')
        if np.max(np.abs(np.diff(d[start:end+1])))>45:flags.append('Distance jump >45 px/frame')
        entry=dict(segment=n,start_row=start,end_row=end,onset_row=onset,peak_row=peak,closure_confirmed_row=stable,near_rows=[s,e],source_start_frame=int(rows[start]['frame']),source_end_frame=int(rows[end]['frame']),frames=end-start+1,pre_action_frames=onset-start,post_closure_frames=end-stable if stable is not None else None,duration_s=t[end]-t[start],distance_column=dc,aperture_column=ac,unit=unit,mm_per_px=scale,review_flags=flags)
        entries.append(entry)
        if flags:reviews.append(entry)
    # Retain and report all missing intervals, including the recording tail.
    missing=[dict(start_row=s,end_row=e,source_start_frame=rows[s]['frame'],source_end_frame=rows[e]['frame'],frames=e-s+1) for s,e in runs(~np.isfinite(d)|~np.isfinite(a))]
    (OUT/'detection_review.json').write_text(json.dumps(dict(suspicious_segments=reviews,missing_intervals=missing,near_candidate_count=len(near)),indent=2),encoding='utf-8')
    print(f'Detected {len(entries)} candidates; {len(reviews)} need review; missing intervals {missing}',flush=True)
    (OUT/'detection_plan.json').write_text(json.dumps(entries,indent=2),encoding='utf-8')
    sheet=np.full((len(entries)*112,4*200,3),255,np.uint8)
    for entry in entries:
        n=entry['segment'];start=entry['start_row'];end=entry['end_row'];dest=OUT/f'grasp_{n:03d}'
        if dest.exists():raise RuntimeError(f'Output exists: {dest}')
        (dest/'frames').mkdir(parents=True)
        segment=rows[start:end+1]
        with (dest/'camera.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=[*fields,'segment_frame','segment_elapsed_s']);w.writeheader()
            for i,r in enumerate(segment):w.writerow(dict(r,segment_frame=i,segment_elapsed_s=float(r['t_camera'])-t[start]))
        shutil.copy2(SOURCE/'settings.json',dest/'settings.json')
        writer=None
        try:
            for r in segment:
                src=SOURCE/'frames'/f"frame_{int(r['frame']):06d}.jpg";target=dest/'frames'/src.name
                shutil.copy2(src,target);assert digest(src)==digest(target)
                im=cv2.imdecode(np.fromfile(target,np.uint8),1)
                if im is None:raise ValueError(str(src))
                if writer is None:
                    h,w=im.shape[:2];writer=cv2.VideoWriter(str(dest/'grasp_video.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),float(settings['fps']),(w,h));assert writer.isOpened()
                writer.write(im)
        finally:
            if writer:writer.release()
        cap=cv2.VideoCapture(str(dest/'grasp_video.mp4'));count=0
        while True:
            ok,_=cap.read()
            if not ok:break
            count+=1
        cap.release()
        with (dest/'camera.csv').open(encoding='utf-8-sig',newline='') as f:saved=list(csv.DictReader(f))
        assert len(saved)==len(segment)==count==len(list((dest/'frames').glob('*.jpg')))
        assert all(all(x[k]==y[k] for k in fields) for x,y in zip(segment,saved))
        entry['validation']=dict(csv_rows=len(saved),image_frames=count,decoded_video_frames=count,original_fields_preserved=True,image_bytes_preserved=True)
        (dest/'segment.json').write_text(json.dumps(entry,indent=2),encoding='utf-8')
        fig,ax=plt.subplots(figsize=(8,6))
        ax.plot(dd[start:end+1],aa[start:end+1],'.-',lw=1,markersize=3)
        ax.scatter(dd[start],aa[start],color='green',label='Start',zorder=3)
        ax.scatter(dd[end],aa[end],color='red',marker='s',label='End',zorder=3)
        ax.set(title=f'grasp_{n:03d} | distance-aperture',xlabel=f'Index MCP distance ({unit})',ylabel=f'Finger aperture ({unit})')
        ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(dest/'distance_aperture.png',dpi=160);plt.close(fig)

        for col,idx in enumerate([start,entry['peak_row'],entry['closure_confirmed_row'] or end,end]):
            im=cv2.imdecode(np.fromfile(SOURCE/'frames'/f"frame_{int(rows[idx]['frame']):06d}.jpg",np.uint8),1)
            sheet[(n-1)*112:(n-1)*112+90,col*200:col*200+120]=cv2.resize(im,(120,90))
            cv2.putText(sheet,f'{n}: {int(rows[idx]["frame"])}',(col*200,(n-1)*112+105),0,.4,(0,0,0),1)
        print(f'{n:03d}: {start}-{end}, {count} frames, flags={entry["review_flags"]}',flush=True)
    cv2.imwrite(str(OUT/'segment_boundaries.jpg'),sheet)
    plots=[('distance_aperture','Index MCP distance ('+unit+')','Aperture ('+unit+')'),('distance_time','Time (s)','Index MCP distance ('+unit+')'),('aperture_time','Time (s)','Aperture ('+unit+')')]
    def draw(ax,k):
        for entry in entries:
            s=entry['start_row'];e=entry['end_row']+1;tt=t[s:e]-t[s];x,y=[(dd[s:e],aa[s:e]),(tt,dd[s:e]),(tt,aa[s:e])][k]
            ax.plot(x,y,lw=.8,alpha=.65,label=str(entry['segment']))
        ax.set_xlabel(plots[k][1]);ax.set_ylabel(plots[k][2]);ax.grid(alpha=.25)
    for k,(name,_,__) in enumerate(plots):
        fig,ax=plt.subplots(figsize=(11,7));draw(ax,k);fig.tight_layout();fig.savefig(OUT/(name+'.png'),dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(22,6))
    for k,ax in enumerate(axs):draw(ax,k)
    fig.suptitle(f'{len(entries)} grasps | raw measurements | actual elapsed timestamps | {unit}')
    fig.tight_layout();fig.savefig(OUT/'combined_overlays.png',dpi=180);plt.close(fig)
    (OUT/'summary.json').write_text(json.dumps(entries,indent=2),encoding='utf-8')
    with (OUT/'crop_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(entries[0]));w.writeheader();w.writerows(entries)
    assert all(digest(SOURCE/name)==h for name,h in originals.items())
    (OUT/'validation.json').write_text(json.dumps(dict(original_file_sha256=originals,originals_unchanged=True,segments=len(entries),all_frame_counts_match=True,all_original_fields_preserved=True,all_copied_images_byte_identical=True),indent=2),encoding='utf-8')
    (OUT/'README.txt').write_text('Primary distance: Euclidean landmark 5 to object center; verified against x/y.\nCalibration from this settings.json: '+str(scale)+' mm/px. Approximate scale at cube top-face height.\nDetection uses raw px: near <55, merge gaps <=8 frames; distant closed rest: distance >230 and aperture <15; backtrack opening to local rest baseline +3 px aperture / -4 px distance, then backtrack until pre-action 10-frame distance range <=4 px and aperture range <=3 px.\nClosure: five frames aperture <= local near-phase 25th percentile +2.5 px, range <=3 px, distance <46 px for closure confirmation plus 10 hold frames; aperture remains <= target+5.\nStart: 10 frames before departure; end: 10 frames after five-frame closure confirmation. No smoothing or normalization. Actual t_camera minus segment first timestamp.\nOriginal CSV fields and frame IDs retained; added segment_frame and segment_elapsed_s. Raw JPEG bytes copied. Video uses settings fps=30; actual timing is in CSV (video is constant rate).\nAll candidates retained, including suspicious candidates; see detection_review.json for flags and all missing intervals.\nAll CSV/image/decoded-video counts and original values verified. Original top-level files SHA256 checked unchanged.\n',encoding='utf-8')
    print('ALL VALIDATED',flush=True)

if __name__=='__main__':main()
