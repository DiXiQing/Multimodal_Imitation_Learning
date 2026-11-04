"""Auditable segmentation for the November 1 black 40mm 21:30 recording with persistent-onset validation; raw measurements unchanged."""
import os
from pathlib import Path
os.environ['MPLCONFIGDIR'] = str(Path(__file__).parent / '.mplconfig')
import csv, json, hashlib, shutil
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plot_trial_distance_aperture import export_distance_aperture

import argparse
from datetime import datetime, timezone, timedelta
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('source',type=Path)
parser.add_argument('--detect-only',action='store_true')
args=parser.parse_args()
SOURCE=args.source.resolve()
OUT = SOURCE/'grasp_segments'

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def main():
    if not args.detect_only:
        if OUT.exists():raise RuntimeError('Output already exists; preserve it')
        OUT.mkdir()
    originals={p.name:digest(p) for p in SOURCE.iterdir() if p.is_file()}
    settings=json.loads((SOURCE/'settings.json').read_text(encoding='utf-8-sig'))
    with (SOURCE/'camera.csv').open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
    def values(col):return np.array([float(r[col]) if r.get(col) else np.nan for r in rows])
    d=values('index_mcp_object_distance');a=values('finger_width');t=values('t_camera')
    scale=(settings.get('pixel_calibration') or {}).get('mm_per_px')
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
    for s,e in runs(d<60):
        if near and s-near[-1][1]-1<=6 and np.all(np.isfinite(d[near[-1][1]+1:s])) and np.all(np.isfinite(a[near[-1][1]+1:s])):near[-1]=(near[-1][0],e)
        else:near.append((s,e))
    entries=[];reviews=[];previous=0
    for n,(s,e) in enumerate(near,1):
        lo=previous;previous=e+1
        invalid_before=np.flatnonzero(~np.isfinite(d[:s+1])|~np.isfinite(a[:s+1]))
        if len(invalid_before):lo=max(lo,int(invalid_before[-1])+1)
        lo=max(lo,s-300)
        # Find the most recent distant, closed posture; do not include prior withdrawal.
        finite=d[lo:s+1][np.isfinite(d[lo:s+1])]
        far=max(100.,float(np.percentile(finite,90))*.80)
        rest_runs=runs((a[lo:s+1]<15)&(d[lo:s+1]>far))
        rest=[lo+j for rs,re in rest_runs for j in range(rs,re+1)]
        # Search quiet local baselines; an early twitch is rejected if the hand
        # subsequently returns to that baseline for >=8 consecutive frames before
        # the main opening peak. Never trim by duration or by opening alone.
        onset_candidates=[]
        main_peak=lo+int(np.nanargmax(a[lo:s+1]))
        for i in range(lo+10, min(s,main_peak)+1):
            bd=float(np.median(d[i-10:i]));ba=float(np.median(a[i-10:i]))
            if not (np.all(np.isfinite(d[i-10:i+5])) and np.all(np.isfinite(a[i-10:i+5]))):continue
            if not (bd>far and np.max(a[i-10:i])<15 and np.ptp(d[i-10:i])<=8 and np.ptp(a[i-10:i])<=4):continue
            active=(d[i:i+5]<bd-6)|(a[i:i+5]>ba+5)
            if np.count_nonzero(active)<3:continue
            returned=(np.abs(d[i:main_peak+1]-bd)<=4)&(a[i:main_peak+1]<=ba+3)
            if any(re-rs+1>=8 for rs,re in runs(returned)):continue
            refined=i
            # Include subtle leading movement immediately preceding the trigger.
            for j in range(max(lo+10,i-6),i+1):
                early=(d[j:j+3]<bd-2.5)|(a[j:j+3]>ba+2.5)
                if np.count_nonzero(early)>=2:refined=j;break
            onset_candidates.append(refined)
        if onset_candidates:onset=min(onset_candidates)
        elif rest_runs:onset=min(s,lo+rest_runs[-1][1]+1)
        else:onset=max(lo,s-30)
        start=max(lo,onset-10)
        peak=onset+int(np.nanargmax(a[onset:s+1]))
        quiet=(onset-start==10 and np.all(np.isfinite(d[start:onset])) and np.all(np.isfinite(a[start:onset])) and np.ptp(d[start:onset])<=8 and np.ptp(a[start:onset])<=4 and np.max(a[start:onset])<15)
        near_aperture=a[max(s,peak+1):e+1]
        target=float(np.nanpercentile(near_aperture,25)) if np.isfinite(near_aperture).any() else np.nan
        candidates=[i for i in range(max(s,peak+1),e-13) if np.all(np.isfinite(a[i:i+15])) and np.max(a[i:i+5])<=target+2.5 and np.ptp(a[i:i+5])<=3 and np.all(d[i:i+15]<50) and np.max(a[i:i+15])<=target+5]
        flags=[]
        # Flag an aborted opening or detection interruption before the main peak.
        for cs,ce in runs(a[onset:peak]<12):
            cs+=onset;ce+=onset
            if cs>onset and ce-cs+1>=8 and np.nanmax(a[onset:cs])>18 and a[peak]>40:
                flags.append('Opening attempt followed by >=8 closed frames before main opening; review repeated action or detection interruption')
                break
        # Retain but flag ambiguous early adjustment followed by another pause.
        opening=next((i for i in range(onset,max(onset,s-2)) if np.all(a[i:i+3]>15)),s)
        paused=[]
        for i in range(onset+5,opening-9):
            if np.all(np.isfinite(d[i:i+10])) and np.all(np.isfinite(a[i:i+10])) and np.ptp(d[i:i+10])<=8 and np.ptp(a[i:i+10])<=4 and np.max(a[i:i+10])<15:paused.append(i)
        if paused:flags.append('Quiet pause between detected departure and main opening; onset requires review')
        if not onset_candidates:flags.append('Persistent departure from a quiet 10-frame baseline not confirmed; onset requires review')
        if not rest:flags.append('No confirmed distant closed rest before action')
        if e-s+1<15:flags.append('Near-object interval shorter than closure+hold confirmation; possible detection artifact or incomplete grasp')
        if not quiet:flags.append('Pre-action 10-frame rest not confirmed; retained for review')
        if candidates:stable=candidates[0]+4;end=stable+10
        else:
            stable=None;end=e;flags.append('No confirmed closure plus 10-frame hold; retained for review')
        if float(np.nanmax(a[start:end+1]))-target<10:flags.append('Insufficient aperture excursion; incomplete action or detection artifact')
        if onset-start!=10:flags.append('Insufficient pre-action margin')
        if np.any(~np.isfinite(d[start:end+1])) or np.any(~np.isfinite(a[start:end+1])):flags.append('Missing measurements')
        if np.any(np.diff(t[start:end+1])<=0):flags.append('Non-increasing timestamps')
        invalid=sum(r.get('valid')!='1' or r.get('index_mcp_distance_valid')!='1' for r in rows[start:end+1])
        if invalid:flags.append(f'{invalid} invalid detections')
        if len(d[start:end+1])>1 and np.nanmax(np.abs(np.diff(d[start:end+1])))>45:flags.append('Distance jump >45 px/frame')
        entry=dict(segment=n,start_row=start,end_row=end,onset_row=onset,peak_row=peak,closure_confirmed_row=stable,near_rows=[s,e],source_start_frame=int(rows[start]['frame']),source_end_frame=int(rows[end]['frame']),frames=end-start+1,pre_action_frames=onset-start,post_closure_frames=end-stable if stable is not None else None,duration_s=t[end]-t[start],distance_column=dc,aperture_column=ac,unit=unit,mm_per_px=scale,review_flags=flags)
        trial_name=datetime.fromtimestamp(float(t[start]),timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')
        if any(Path(x['relative_folder']).name==trial_name for x in entries):raise ValueError('Two candidates start in the same second; review boundaries before export, no suffix added')
        entry['trial_name']=trial_name
        entry['relative_folder']=('review_segments/' if flags else '')+trial_name
        entries.append(entry)
        if flags:reviews.append(entry)
    if args.detect_only:
        print('DETECTED',SOURCE.name,'candidates',len(entries),'review',len(reviews))
        for x in entries:print(x['segment'],x['start_row'],x['end_row'],round(x['duration_s'],2),x['review_flags'])
        return
    # Retain and report all missing intervals, including the recording tail.
    missing=[dict(start_row=s,end_row=e,source_start_frame=rows[s]['frame'],source_end_frame=rows[e]['frame'],frames=e-s+1) for s,e in runs(~np.isfinite(d)|~np.isfinite(a))]
    (OUT/'detection_review.json').write_text(json.dumps(dict(suspicious_segments=reviews,missing_intervals=missing,near_candidate_count=len(near)),indent=2),encoding='utf-8')
    print(f'Detected {len(entries)} candidates; {len(reviews)} need review; missing intervals {missing}',flush=True)
    sheet=np.full((len(entries)*112,4*200,3),255,np.uint8)
    for entry in entries:
        n=entry['segment'];start=entry['start_row'];end=entry['end_row'];dest=OUT/entry['relative_folder']
        # Refresh generated exports after adjusting start boundaries.
        (dest/'frames').mkdir(parents=True,exist_ok=False)
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
        export_distance_aperture(saved, dest, unit)
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
    (OUT/'README.txt').write_text('Primary distance: Euclidean landmark 5 to object center; verified against x/y.\nCalibration from this settings.json: '+str(scale)+' mm/px (None means uncalibrated; plots and primary fields are px, no assumed ratio).\nDetection uses raw px: near <60, merge finite gaps <=6 frames, never cross missing intervals, lookback <=300 frames; distant closed rest: distance >80% local distant-distance 90th percentile (minimum100px) and aperture <15; Persistent onset: quiet local 10-frame baseline; 3/5 frames depart by distance -6px or aperture +5px; reject if returning within baseline distance 4px/aperture +3px for 8 frames before main opening peak; backtrack up to6 frames for smaller 2.5px departure, retain10 prior frames. No fixed-duration trimming. Flag if their distance range >8px, aperture range >4px, or aperture reaches 15px.\nClosure: five frames aperture <= local near-phase 25th percentile +2.5 px, range <=3 px, distance <50 px for closure confirmation plus 10 hold frames; aperture remains <= target+5.\nStart: 10 frames before departure; end: 10 frames after five-frame closure confirmation. Raw measurements unchanged; per-trial distance-aperture plots use initial-distance normalization, aperture remains in recorded units; no time normalization or smoothing. Actual t_camera minus segment first timestamp.\nOriginal CSV fields and frame IDs retained; added segment_frame and segment_elapsed_s. Raw JPEG bytes copied. Video uses settings fps=30; actual timing is in CSV (video is constant rate).\nAll candidates retained, including suspicious candidates; see detection_review.json for flags and all missing intervals.\nAll CSV/image/decoded-video counts and original values verified. Original top-level files SHA256 checked unchanged.\n',encoding='utf-8')
    print('ALL VALIDATED',flush=True)
    with (OUT/'README.txt').open('a',encoding='utf-8') as f:f.write('\nTrial folders use first-frame Asia/Tokyo timestamps YYYYMMDD_HHMMSS, without fractions or suffixes. Unflagged segments are at top level. All flagged candidates are retained separately under review_segments/; flags do not by themselves establish actual grasp failure. summary.json lists relative_folder for each candidate.\n')

if __name__=='__main__':main()

