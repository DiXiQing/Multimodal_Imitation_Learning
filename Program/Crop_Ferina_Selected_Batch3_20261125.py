import csv,json,shutil,sys,hashlib
from pathlib import Path
from datetime import datetime
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tool'))
from plot_trial_distance_aperture import export_distance_aperture
base=ROOT/'Data/1118_ferina/4_2';backup=base/'backup'/('manual_crop_'+datetime.now().strftime('%Y%m%d_%H%M%S'));backup.mkdir(parents=True)
plan={'20261125_124811':(0,1.98),'20261125_124802':(0,1.75),'20261125_124759':(0,1.70),'20261125_124630':(0,1.55),'20261125_124830':(.18,1.42),'20261125_124451':(0,1.35)}
fig,axes=plt.subplots(3,2,figsize=(12,10));results=[]
for ax,(name,(lo,hi)) in zip(axes.flat,plan.items()):
 folder=base/name
 with (folder/'camera.csv').open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
 t=np.array([float(r['t_camera'])-float(rows[0]['t_camera']) for r in rows]);indices=np.flatnonzero((t>=lo)&(t<=hi));assert len(indices)>1
 first,last=int(indices[0]),int(indices[-1]);kept=[dict(r) for r in rows[first:last+1]]
 stage=backup/(name+'_prepared');stage.mkdir();(stage/'frames').mkdir()
 origin=float(kept[0]['t_camera'])
 for i,r in enumerate(kept):
  r['segment_frame']=str(i);r['segment_elapsed_s']=str(float(r['t_camera'])-origin)
  src=folder/'frames'/f"frame_{int(r['frame']):06d}.jpg";dst=stage/'frames'/src.name;shutil.copy2(src,dst);assert src.read_bytes()==dst.read_bytes()
 with (stage/'camera.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(kept)
 shutil.copy2(folder/'settings.json',stage/'settings.json')
 cap=cv2.VideoCapture(str(folder/'grasp_video.mp4'));assert int(cap.get(7))==len(rows)
 writer=cv2.VideoWriter(str(stage/'grasp_video.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),cap.get(5),(int(cap.get(3)),int(cap.get(4))));assert writer.isOpened()
 for i in range(last+1):
  ok,im=cap.read();assert ok
  if i>=first:writer.write(im)
 cap.release();writer.release();check=cv2.VideoCapture(str(stage/'grasp_video.mp4'));assert int(check.get(7))==len(kept);check.release()
 meta=json.loads((folder/'segment.json').read_text(encoding='utf-8-sig'));old=dict(meta)
 start=int(meta.get('start_row',rows[0]['frame']))+first;end=start+len(kept)-1
 meta.update(start_row=start,end_row=end,source_start_frame=int(kept[0]['frame']),source_end_frame=int(kept[-1]['frame']),frames=len(kept),duration_s=float(kept[-1]['t_camera'])-origin,relative_folder=name)
 onset=meta.get('onset_row');meta['pre_action_frames']=onset-start if onset is not None and start<=onset<=end else None
 stable=meta.get('closure_confirmed_row');meta['post_closure_frames']=end-stable if stable is not None and start<=stable<=end else None
 meta['manual_crop']=dict(original_time_range=[float(t[first]),float(t[last])],backup=str(backup/name),note='User screenshot selection; original acquisition measurements unchanged; onset/closure annotations inherited and may require review.')
 meta['validation']=dict(csv_rows=len(kept),image_frames=len(kept),decoded_video_frames=len(kept),original_fields_preserved=True,image_bytes_preserved=True)
 (stage/'segment.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');export_distance_aperture(kept,stage,'mm')
 yy=np.array([float(r['finger_width_mm']) for r in rows]);ax.plot(t,yy,color='0.75',label='Before');ax.plot(t[first:last+1],yy[first:last+1],color='#287db5',lw=2,label='Retained');ax.axvline(t[first],color='#db7356',ls='--');ax.axvline(t[last],color='#db7356',ls='--');ax.set(title=name,xlabel='Original segment time (s)',ylabel='Finger aperture (mm)');ax.grid(alpha=.2);ax.legend()
 fig2,aa=plt.subplots(figsize=(7,4));aa.plot([float(r['segment_elapsed_s']) for r in kept],[float(r['finger_width_mm']) for r in kept]);aa.set(xlabel='Time (s)',ylabel='Finger aperture (mm)');aa.grid(alpha=.2);fig2.tight_layout();fig2.savefig(stage/'aperture_time.png',dpi=160);plt.close(fig2)
 assert all(all(r[k]==q[k] for k in fields if k not in ['segment_frame','segment_elapsed_s']) for r,q in zip(rows[first:last+1],kept))
 assert folder.resolve().parent==base.resolve() and (backup/name).resolve().is_relative_to(base.resolve())
 folder.rename(backup/name)
 try:stage.rename(folder)
 except Exception:(backup/name).rename(folder);raise
 result=dict(trial=name,old_frames=len(rows),new_frames=len(kept),original_start_s=float(t[first]),original_end_s=float(t[last]),new_duration_s=meta['duration_s'],backup=str(backup/name));results.append(result);print(result,flush=True)
fig.tight_layout();fig.savefig(backup/'crop_comparison.png',dpi=160);plt.close(fig)
(backup/'manifest.json').write_text(json.dumps(results,indent=2),encoding='utf-8');print('OUTPUT',backup)
