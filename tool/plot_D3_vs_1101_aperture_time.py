"""Overlay actual elapsed time vs raw fingertip aperture for D3 and 1101."""
from pathlib import Path
import csv,json,re,os,hashlib
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).parent/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'Report/D3_vs_1101_20mm_aperture_time';OUT.mkdir(parents=True,exist_ok=True)
sources=[('D3',BASE/'Data/D3',['Train','Validation'],'#287db5'),('1101',BASE/'Data/1101_D/20mm',[''],'#e58b26')]
trials=[];metadata=[]
for name,root,parts,color in sources:
 seen=set()
 for sub in parts:
  for folder in sorted((root/sub).iterdir()):
   if not folder.is_dir() or not (folder/'camera.csv').is_file() or not (folder/'segment.json').is_file():continue
   if name=='1101' and not re.fullmatch(r'\d{8}_\d{6}',folder.name):continue
   if folder.name in seen:raise ValueError(f'Duplicate trial {name}/{folder.name}')
   seen.add(folder.name)
   path=folder/'camera.csv';before=hashlib.sha256(path.read_bytes()).hexdigest()
   with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
   t=np.array([float(r['t_camera']) for r in rows]);a=np.array([float(r['finger_width']) if r.get('finger_width') else np.nan for r in rows]);frame=np.array([int(r['frame']) for r in rows])
   assert np.isfinite(t).all() and np.all(np.diff(t)>0)
   a[np.array([r.get('valid')!='1' for r in rows])]=np.nan
   t=t-t[0]
   breaks=np.flatnonzero((np.diff(frame)!=1)|(np.diff(t)>.1))+1
   plot_t=np.insert(t,breaks,np.nan);plot_a=np.insert(a,breaks,np.nan)
   trials.append(dict(dataset=name,color=color,t=plot_t,a=plot_a))
   settings=json.loads((folder/'settings.json').read_text(encoding='utf-8-sig'))
   metadata.append(dict(dataset=name,trial=folder.name,source=str(path),split=sub or 'active root',frames=len(rows),duration_s=float(t[-1]),initial_aperture_px=float(a[0]),peak_aperture_px=float(np.nanmax(a)),final10_aperture_mean_px=float(np.nanmean(a[-10:])),recorded_size_cm=settings.get('size_cm'),image_width=settings.get('width'),image_height=settings.get('height'),csv_sha256=before))
   assert hashlib.sha256(path.read_bytes()).hexdigest()==before
fig,ax=plt.subplots(figsize=(8.5,5.5),dpi=240)
for g in trials:ax.plot(g['t'],g['a'],color=g['color'],alpha=.24,lw=.8)
counts={name:sum(g['dataset']==name for g in trials) for name,_,_,_ in sources}
ax.legend(handles=[Line2D([0],[0],color=color,lw=2,label=f'{name} (n={counts[name]})') for name,_,_,color in sources],loc='upper right',fontsize=11)
ax.set_xlabel('Time from clip start (s)',fontsize=12);ax.set_ylabel('Finger aperture (px)',fontsize=12)
ax.set_xlim(left=0);ax.set_ylim(bottom=0);ax.grid(alpha=.17);fig.tight_layout()
for ext in ['png','svg','pdf']:fig.savefig(OUT/f'D3_vs_1101_aperture_time_overlay.{ext}',bbox_inches='tight')
plt.close(fig)
with (OUT/'per_trial_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(metadata[0]));w.writeheader();w.writerows(metadata)
info=dict(counts=counts,colors={name:color for name,_,_,color in sources},aperture='finger_width (px), raw measured fingertip separation',time='Actual t_camera minus each clip first timestamp; retained pre/post margins included',normalization='None: no time or aperture normalization',smoothing=False,missing='Invalid measurements and frame/time gaps shown as discontinuities',exclusions='backup, review_segments, Result, duplicate speed folders',caveat='D3 recording labels say 40mm, 540x480; 1101 actual object confirmed20mm, 540x380. D3 lacks mm calibration. Pixel amplitude differences are not directly comparable physical aperture differences.',source_files_unchanged=True,summary={name:dict(duration_range_s=[min(r['duration_s'] for r in metadata if r['dataset']==name),max(r['duration_s'] for r in metadata if r['dataset']==name)],peak_aperture_median_px=float(np.median([r['peak_aperture_px'] for r in metadata if r['dataset']==name]))) for name in counts})
(OUT/'plot_information.json').write_text(json.dumps(info,indent=2),encoding='utf-8');print(json.dumps(info,indent=2))
