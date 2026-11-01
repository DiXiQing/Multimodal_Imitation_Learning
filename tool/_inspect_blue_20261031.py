"""Auditable segmentation for the blue 3 cm recording; raw measurements unchanged."""
import os
from pathlib import Path
os.environ['MPLCONFIGDIR'] = str(Path(__file__).parent / '.mplconfig')
import csv, json, hashlib, shutil
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SOURCE = Path(__file__).resolve().parents[1]/'Data/TopView/TRIAL_Color_blue_Size_3cm_20261031_190109_221153'
OUT = SOURCE/'grasp_segments'

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def main():
    pass
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
    for s,e in runs(d<54):
        if near and s-near[-1][1]-1<=8:near[-1]=(near[-1][0],e)
        else:near.append((s,e))
    entries=[];reviews=[];previous=0
    for n,(s,e) in enumerate(near,1):
        lo=previous;previous=e+1
        peak=lo+int(np.nanargmax(a[lo:s+1]))
        # Last closed, distant rest before opening; adaptive local baseline.
        rest=[i for i in range(lo+10,peak) if np.all(a[i-10:i+1]<15) and np.all(d[i-10:i+1]>225)]
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
        if np.any(~np.isfinite(d[start:end+1])) or np.any(~np.isfinite(a[start:end+1])):flags.append('Missing measurements')
        if np.any(np.diff(t[start:end+1])<=0):flags.append('Non-increasing timestamps')
        invalid=sum(r.get('valid')!='1' or r.get('index_mcp_distance_valid')!='1' for r in rows[start:end+1])
        if invalid:flags.append(f'{invalid} invalid detections')
        if np.max(np.abs(np.diff(d[start:end+1])))>45:flags.append('Distance jump >45 px/frame')
        entry=dict(segment=n,start_row=start,end_row=end,onset_row=onset,peak_row=peak,closure_confirmed_row=stable,near_rows=[s,e],source_start_frame=int(rows[start]['frame']),source_end_frame=int(rows[end]['frame']),frames=end-start+1,pre_action_frames=onset-start,post_closure_frames=end-stable if stable is not None else None,duration_s=t[end]-t[start],distance_column=dc,aperture_column=ac,unit=unit,mm_per_px=scale,review_flags=flags)
        entries.append(entry)
        if flags:reviews.append(entry)
    print('CANDIDATES',len(entries),'FLAGGED',len(reviews))
    for e in entries:print(e['segment'],e['start_row'],e['end_row'],round(e['duration_s'],2),e['review_flags'])

main()
