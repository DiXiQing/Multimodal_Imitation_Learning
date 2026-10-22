"""Trim reviewed terminal aperture rises, preserving full backups and original retained values."""
from pathlib import Path
from datetime import datetime
import csv,json,hashlib,zipfile
import numpy as np,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]/'New/Data/D2/slow'
def read(p):
 with p.open(encoding='utf-8-sig') as f:r=csv.DictReader(f);return list(r.fieldnames),list(r)
def write(p,fields,rows):
 with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
plans=[];alltrials=[]
for p in sorted(ROOT.glob('*/camera.csv')):
 fields,rows=read(p);m=json.loads((p.parent/'segment.json').read_text(encoding='utf-8-sig'));a=np.array([float(r['finger_width']) for r in rows]);stable=int(m['stable'])-int(m['start'])
 sm=np.array([np.median(a[max(stable,i-1):min(len(a),i+2)]) for i in range(stable,len(a))])
 minimum=int(np.argmin(sm));level=float(sm[minimum]);rise=float(np.median(a[-2:])-level);cut=None
 if rise>=3 and minimum+2<len(sm):
  for j in range(minimum+1,len(sm)-1):
   if sm[j]>level+1.5 and sm[j+1]>level+1.5:cut=stable+j;break
 if cut is not None and cut>=stable+3 and cut<len(rows):
  plans.append(dict(folder=p.parent,fields=fields,rows=rows,meta=m,cut=cut,level=level,rise=rise))
 alltrials.append((p.parent.name,a,cut))
print('Planned',len(plans),'trials;',sum(len(p['rows'])-p['cut'] for p in plans),'frames',flush=True)
backup=ROOT/'backup'/f'tail_trim_{datetime.now():%Y%m%d_%H%M%S}';backup.mkdir(parents=True)
hashes={}
with zipfile.ZipFile(backup/'original_trials.zip','w',compression=zipfile.ZIP_STORED) as z:
 for plan in plans:
  for p in plan['folder'].rglob('*'):
   if p.is_file():
    rel=p.relative_to(ROOT).as_posix();raw=p.read_bytes();hashes[rel]=hashlib.sha256(raw).hexdigest();z.writestr(rel,raw)
with zipfile.ZipFile(backup/'original_trials.zip') as z:
 for name,digest in hashes.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
(backup/'sha256.json').write_text(json.dumps(hashes,indent=2))
print('Backup verified',flush=True)
log=[]
for plan in plans:
 folder=plan['folder'];rows=plan['rows'];cut=plan['cut'];kept=rows[:cut];m=plan['meta'];videos=list(folder.glob('*.mp4'));temporary=[]
 for video in videos:
  cap=cv2.VideoCapture(str(video));assert int(cap.get(7))==len(rows),(video,cap.get(7),len(rows));fps=cap.get(5);size=(int(cap.get(3)),int(cap.get(4)));temp=video.with_name(video.stem+'_trim_pending.mp4');writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),fps,size);assert writer.isOpened()
  try:
   for i in range(cut):
    ok,image=cap.read();assert ok;writer.write(image)
  finally:writer.release();cap.release()
  check=cv2.VideoCapture(str(temp));assert int(check.get(7))==cut;check.release();temporary.append((temp,video))
 # Resolve exact image paths inside this trial before removing only cropped tail frames.
 removed=[]
 for r in rows[cut:]:
  p=folder/'frames'/f'frame_{int(r["frame"]):06d}.jpg'
  if p.exists():
   assert p.resolve().is_relative_to(folder.resolve());removed.append(p)
 write(folder/'camera.csv',plan['fields'],kept)
 for temp,video in temporary:temp.replace(video)
 for p in removed:p.unlink()
 oldend=m['end'];m.update(end=int(m['start'])+cut-1,frames=cut,post_stable_frames=int(m['start'])+cut-1-int(m['stable']),source_end_frame=int(kept[-1]['frame']),duration_s=float(kept[-1]['t_camera'])-float(kept[0]['t_camera']),invalid_frames=sum(r.get('valid')!='1' for r in kept))
 m['tail_trim']=dict(original_end=oldend,original_frames=len(rows),removed_frames=len(rows)-cut,backup=str(backup),reason='Sustained terminal aperture re-opening after stable hold',rule='Last-two-frame median >=3 px above minimum 3-point median after stable; trim at first two smoothed points >minimum+1.5 px; retain at least 3 stable frames')
 (folder/'segment.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
 _,saved=read(folder/'camera.csv');assert saved==kept
 log.append(dict(trial=folder.name,original_frames=len(rows),retained_frames=cut,removed_frames=len(rows)-cut,hold_minimum_px=plan['level'],tail_rise_px=plan['rise'],last_retained_source_frame=int(kept[-1]['frame'])))
 print(folder.name,'removed',len(rows)-cut,flush=True)
write(backup/'trim_manifest.csv',list(log[0]),log)
write(ROOT/'tail_trim_manifest.csv',list(log[0]),log)
# Update matching rows in any local crop summary without touching backup copies.
summary=ROOT/'crop_summary.csv'
if summary.exists():
 fields,entries=read(summary)
 for entry in entries:
  folder=ROOT/entry.get('folder_name','')
  if any(p['folder']==folder for p in plans):
   meta=json.loads((folder/'segment.json').read_text())
   for k in fields:
    if k in meta:entry[k]=meta[k]
 write(summary,fields,entries)
fig,axes=plt.subplots(int(np.ceil(len(alltrials)/6)),6,figsize=(18,2.4*int(np.ceil(len(alltrials)/6))),squeeze=False)
for ax,(name,a,cut) in zip(axes.flat,alltrials):
 ax.plot(np.arange(len(a)),a,color='#bbbbbb',lw=1,label='Before')
 ax.plot(np.arange(cut or len(a)),a[:cut],color='#3485b8',lw=1,label='Retained')
 if cut is not None:ax.axvline(cut-.5,color='#d9534f',lw=.8)
 ax.set_title(name[-6:],fontsize=9);ax.grid(alpha=.2)
for ax in list(axes.flat)[len(alltrials):]:ax.axis('off')
axes.flat[0].legend(fontsize=7);fig.tight_layout();fig.savefig(ROOT/'tail_trim_before_after.png',dpi=150);plt.close(fig)
print('DONE',len(plans),'trials',sum(r['removed_frames'] for r in log),'frames; backup:',backup,flush=True)
