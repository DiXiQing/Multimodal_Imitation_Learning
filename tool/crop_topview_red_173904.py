"""Split the 20261017 top-view recording into approach-and-grasp segments."""
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
import shutil
from pathlib import Path
import cv2
import numpy as np

SOURCE = Path(__file__).resolve().parents[1]/'Data/TopView/TRIAL_Color_red_Size_2cm_20261017_173904_324695'
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
    edges=np.diff(np.r_[False,distance<35,False].astype(int))
    runs=list(zip(np.where(edges==1)[0],np.where(edges==-1)[0]-1))
    ranges=[];prev_end=0
    for n,(near_start,near_end) in enumerate(runs,1):
        near_start=int(near_start);near_end=int(near_end)
        if near_end-near_start<10:continue
        look=max(prev_end+1,near_start-55)
        peak=look+int(np.argmax(aperture[look:near_start+1]))
        resting=float(np.quantile(aperture[look:peak],.15))
        below=np.flatnonzero(aperture[look:peak]<=resting+3)
        onset=look+int(below[-1])+1 if len(below) else look
        start=max(look,onset-10)
        lo=near_start+5;hi=near_end-5
        minimum=lo+int(np.argmin(aperture[lo:hi+1]))
        grasp_width=float(np.median(aperture[max(lo,minimum-2):minimum+3]))
        stable_candidates=np.flatnonzero(aperture[lo:minimum+1]<=grasp_width+3)
        stable=lo+int(stable_candidates[0]) if len(stable_candidates) else minimum
        end=minimum
        for i in range(minimum+1,near_end-1):
            if np.all(aperture[i:i+3]>grasp_width+3):break
            end=i
        segment=rows[start:end+1]
        invalid=sum(r['valid']!='1' for r in segment)
        entry=dict(segment=n,start=start,onset=onset,peak=peak,stable=stable,end=end,
            frames=len(segment),duration_s=float(segment[-1]['t_camera'])-float(segment[0]['t_camera']),
            invalid_frames=invalid,source_start_frame=int(segment[0]['frame']),source_end_frame=int(segment[-1]['frame']))
        ranges.append(entry);prev_end=near_end
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
    for p,h in hashes.items():assert hashlib.sha256(p.read_bytes()).hexdigest()==h
    (OUT/'README.txt').write_text('15 approach-and-grasp segments for red 2 cm object. End before release/withdrawal.\nOriginal CSV values and source frame IDs preserved; segment_frame and segment_elapsed_s added.\nBoundary detection uses median-smoothed signals only; exported measurements are unchanged.\nMissing values are preserved. Original images are copied byte-for-byte. Videos are 30 FPS; use CSV timestamps for actual time.\nSee crop_summary.csv and segment_boundaries.jpg. Original recording remains unchanged.\n',encoding='utf-8')
    print('VERIFIED: CSV values, frame bytes, decoded video counts, original CSV/video/settings hashes.',flush=True)

    import subprocess, sys
    subprocess.run([sys.executable, '-s', str(Path(__file__).resolve().parents[1]/'New/Program/plot_relationship.py'), '--source', str(OUT)], check=True)

if __name__=='__main__':main()
