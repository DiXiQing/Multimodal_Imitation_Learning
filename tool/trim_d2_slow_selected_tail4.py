from pathlib import Path
import csv,json,zipfile,hashlib
from datetime import datetime
import cv2
root=Path('New/Data/D2/slow').resolve();suffixes='162530 162527 162518 162454 162457 162509 162418 162421 162434 162412 162406 162403 162358 162354 162336'.split();plans=[]
for suffix in suffixes:
 matches=[p for p in root.glob('Black_*') if p.is_dir() and p.name.endswith('_'+suffix)];assert len(matches)==1,(suffix,matches)
 folder=matches[0]
 with (folder/'camera.csv').open(encoding='utf-8-sig') as f:r=csv.DictReader(f);fields=list(r.fieldnames);rows=list(r)
 assert len(rows)>4
 plans.append((folder,fields,rows))
backup=root/'backup'/f'manual_tail4_{datetime.now():%Y%m%d_%H%M%S}';backup.mkdir(parents=True)
hashes={}
with zipfile.ZipFile(backup/'original_trials.zip','w',compression=zipfile.ZIP_STORED) as z:
 for folder,_,_ in plans:
  for p in folder.rglob('*'):
   if p.is_file():
    name=p.relative_to(root).as_posix();data=p.read_bytes();hashes[name]=hashlib.sha256(data).hexdigest();z.writestr(name,data)
with zipfile.ZipFile(backup/'original_trials.zip') as z:
 for name,h in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
(backup/'sha256.json').write_text(json.dumps(hashes,indent=2));print('Backup verified',flush=True)
log=[]
for folder,fields,rows in plans:
 kept=rows[:-4];n=len(kept);pending=[]
 for video in folder.glob('*.mp4'):
  cap=cv2.VideoCapture(str(video));assert int(cap.get(7))==len(rows)
  temp=video.with_name(video.stem+'_trim_pending.mp4');writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),cap.get(5),(int(cap.get(3)),int(cap.get(4))));assert writer.isOpened()
  try:
   for _ in range(n):ok,frame=cap.read();assert ok;writer.write(frame)
  finally:cap.release();writer.release()
  check=cv2.VideoCapture(str(temp));assert int(check.get(7))==n;check.release();pending.append((temp,video))
 with (folder/'camera.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(kept)
 for temp,video in pending:temp.replace(video)
 for r in rows[-4:]:
  p=folder/'frames'/f'frame_{int(r["frame"]):06d}.jpg'
  if p.exists():assert p.resolve().is_relative_to(folder);p.unlink()
 meta=json.loads((folder/'segment.json').read_text(encoding='utf-8-sig'));meta.update(end=int(meta['start'])+n-1,frames=n,post_stable_frames=int(meta['start'])+n-1-int(meta['stable']),source_end_frame=int(kept[-1]['frame']),duration_s=float(kept[-1]['t_camera'])-float(kept[0]['t_camera']),invalid_frames=sum(r.get('valid')!='1' for r in kept))
 meta.setdefault('manual_tail_trims',[]).append(dict(removed_frames=4,previous_frames=len(rows),backup=str(backup),reason='User-selected four-frame tail trim'))
 (folder/'segment.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
 with (folder/'camera.csv').open(encoding='utf-8-sig') as f:assert list(csv.DictReader(f))==kept
 log.append(dict(trial=folder.name,original_frames=len(rows),retained_frames=n,removed_frames=4));print(folder.name,len(rows),'->',n,flush=True)
for p in [backup/'trim_manifest.csv',root/'manual_tail4_manifest.csv']:
 with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(log[0]));w.writeheader();w.writerows(log)
summary=root/'crop_summary.csv'
if summary.exists():
 with summary.open(encoding='utf-8-sig') as f:r=csv.DictReader(f);fields=list(r.fieldnames);entries=list(r)
 for row in entries:
  if row.get('folder_name') in {x['trial'] for x in log}:
   meta=json.loads((root/row['folder_name']/'segment.json').read_text())
   for k in fields:
    if k in meta:row[k]=meta[k]
 with summary.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(entries)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(3,5,figsize=(16,8))
for ax,(folder,_,rows) in zip(axes.flat,plans):
 a=[float(r['finger_width']) for r in rows];ax.plot(range(1,len(a)+1),a,color='#bbbbbb',label='Before');ax.plot(range(1,len(a)-3),a[:-4],color='#3485b8',label='Retained');ax.axvline(len(a)-3.5,color='#d9534f',lw=.8);ax.set_title(folder.name[-6:]);ax.grid(alpha=.2)
axes.flat[0].legend(fontsize=8);fig.tight_layout();fig.savefig(root/'manual_tail4_before_after.png',dpi=180);plt.close(fig)
print('DONE: 15 trials, 60 frames. Backup:',backup)
