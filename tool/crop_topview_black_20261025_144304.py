"""Split the 20261025 top-view recording into approach-and-grasp segments."""
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
import shutil
from pathlib import Path
import cv2
import numpy as np

SOURCE = Path(__file__).resolve().parents[1]/'Data/TopView/TRIAL_Color_black_Size_4cm_20261025_144304_350412'
OUT = SOURCE/'grasp_segments'

def main():
    OUT.mkdir(exist_ok=True)
    originals = [SOURCE/'camera.csv', SOURCE/'recorded_video.mp4', SOURCE/'settings.json']
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in originals}
    with (SOURCE/'camera.csv').open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f); fields=reader.fieldnames; rows=list(reader)
    def smooth(col, radius):
        a=np.array([float(r[col]) if r[col] else np.nan for r in rows])
        good=np.isfinite(a)
        a=np.interp(np.arange(len(a)),np.flatnonzero(good),a[good])
        return np.array([np.median(a[max(0,i-radius):i+radius+1]) for i in range(len(a))])
    distance=smooth('hand_object_distance',3); aperture=smooth('finger_width',1)
    edges=np.diff(np.r_[False,aperture>20,False].astype(int))
    runs=list(zip(np.where(edges==1)[0],np.where(edges==-1)[0]-1))
    ranges=[]
    for near_start,near_end in runs:
        near_start=int(near_start);near_end=int(near_end)
        if near_end-near_start<12 or near_start<12 or near_end>=len(rows)-2:continue
        peak=near_start+int(np.argmax(aperture[near_start:near_end+1]))
        if near_end-peak<6:continue
        onset=near_start-1;start=max(0,onset-12)
        grasp_width=float(np.percentile(aperture[peak+1:near_end+1],25))
        stable=None
        for i in range(peak+1,near_end-3):
            if np.max(abs(aperture[i:i+5]-grasp_width))<=5:
                stable=i;break
        if stable is None:continue
        end=min(stable+20,near_end)
        for j in range(stable+5,end):
            if aperture[j]>grasp_width+8 and aperture[j+1]>grasp_width+8:
                end=j-1;break
        segment=rows[start:end+1]
        raw_d=np.array([float(r['hand_object_distance']) if r['hand_object_distance'] else np.nan for r in segment])
        flagged=bool(np.any(abs(np.diff(raw_d))>100) or np.nanmedian(raw_d[stable-start:])>100)
        entry=dict(segment=len(ranges)+1,start=start,onset=onset,peak=peak,stable=stable,end=end,
            frames=len(segment),pre_onset_frames=onset-start,post_stable_frames=end-stable,
            boundary_rule='Aperture >20 px action run; stable: five frames within 5 px of post-peak 25th percentile; 12 pre-onset and up to 20 post-stable frames. Distance not used because object tracking jumps.',
            duration_s=float(segment[-1]['t_camera'])-float(segment[0]['t_camera']),
            invalid_frames=sum(r['valid']!='1' for r in segment),distance_tracking_review=flagged,
            source_start_frame=int(segment[0]['frame']),source_end_frame=int(segment[-1]['frame']))
        ranges.append(entry)
    print(f'Detected {len(ranges)} grasp segments.',flush=True)
    sheet=np.full((len(ranges)*150,4*240,3),255,np.uint8)
    for row,entry in enumerate(ranges):
        metadata=json.loads((SOURCE/'settings.json').read_text(encoding='utf-8-sig'))
        stamp=datetime.fromtimestamp(float(rows[entry['start']]['t_camera']),timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')
        dest=OUT/f"{metadata.get('object_color','black').capitalize()}_{metadata['size_cm']:g}cm_{stamp}"
        entry['folder_name']=dest.name
        if dest.exists():raise RuntimeError(f'Already exists: {dest}')
        (dest/'frames').mkdir(parents=True)
        segment=rows[entry['start']:entry['end']+1]
        outfields=['segment_frame','segment_elapsed_s',*fields]
        with (dest/'camera.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=outfields);writer.writeheader()
            for i,r in enumerate(segment):
                writer.writerow(dict(segment_frame=i,segment_elapsed_s=float(r['t_camera'])-float(segment[0]['t_camera']),**r))
        shutil.copy2(SOURCE/'settings.json',dest/'settings.json')
        (dest/'segment.json').write_text(json.dumps(entry,indent=2),encoding='utf-8')
        video=None
        try:
            for r in segment:
                src=SOURCE/'frames'/f'frame_{int(r["frame"]):06d}.jpg'
                raw=src.read_bytes();im=cv2.imdecode(np.frombuffer(raw,np.uint8),1)
                if im is None:raise RuntimeError(f'Cannot decode {src}')
                target=dest/'frames'/src.name;target.write_bytes(raw)
                assert hashlib.sha256(raw).digest()==hashlib.sha256(target.read_bytes()).digest()
                if video is None:
                    h,w=im.shape[:2];video=cv2.VideoWriter(str(dest/'grasp_video.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(w,h))
                    if not video.isOpened():raise RuntimeError('Video open failed')
                video.write(im)
        finally:
            if video is not None:video.release()
        cap=cv2.VideoCapture(str(dest/'grasp_video.mp4'));count=0
        while True:
            ok,_=cap.read()
            if not ok:break
            count+=1
        cap.release()
        assert count==len(segment)==len(list((dest/'frames').glob('*.jpg')))
        with (dest/'camera.csv').open(encoding='utf-8-sig',newline='') as f: saved=list(csv.DictReader(f))
        for original,copy in zip(segment,saved):
            assert all(original[k]==copy[k] for k in fields)
        for col,key in enumerate(['start','peak','stable','end']):
            fidx=int(rows[entry[key]]['frame'])
            im=cv2.imdecode(np.fromfile(SOURCE/'frames'/f'frame_{fidx:06d}.jpg',np.uint8),1)
            im=cv2.resize(im,(230,115));sheet[row*150:row*150+115,col*240:col*240+230]=im
            cv2.putText(sheet,f'{entry["segment"]:02d} {key} {fidx}',(col*240+4,row*150+138),0,.45,(0,0,0),1)
        print(f'Exported {dest.name}: {entry["start"]}-{entry["end"]}, {count} frames, invalid={entry["invalid_frames"]}',flush=True)
    with (OUT/'crop_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(ranges[0]));writer.writeheader();writer.writerows(ranges)
    cv2.imwrite(str(OUT/'segment_boundaries.jpg'),sheet)
    with (OUT/'distance_tracking_review.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(ranges[0]));writer.writeheader();writer.writerows(r for r in ranges if r['distance_tracking_review'])
    for p,h in hashes.items():assert hashlib.sha256(p.read_bytes()).hexdigest()==h
    (OUT/'README.txt').write_text(f'{len(ranges)} approach-and-grasp segments for black 4 cm object. 12-frame pre-onset and 20-frame post-stable margins (capped before withdrawal).\nOriginal CSV values and source frame IDs preserved; segment_frame and segment_elapsed_s added.\nBoundaries use median-smoothed aperture only because object-center jumps corrupt distance in some trials. See distance_tracking_review.csv; flags are review aids, not repaired measurements. Exported measurements are unchanged.\nMissing values are preserved. Original images are copied byte-for-byte. Videos are 30 FPS; use CSV timestamps for actual time.\nSee crop_summary.csv and segment_boundaries.jpg. Original recording remains unchanged.\n',encoding='utf-8')
    print('VERIFIED: CSV values, frame bytes, decoded video counts, original CSV/video/settings hashes.',flush=True)


if __name__=='__main__':main()
