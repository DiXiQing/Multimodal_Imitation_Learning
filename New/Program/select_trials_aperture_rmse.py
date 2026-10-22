"""Select trials by leave-one-out aperture RMSE against a fixed original cohort."""
from pathlib import Path
from datetime import datetime
import csv,json,hashlib,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,default=ROOT/'New/Data/D2/fast');parser.add_argument('--keep',type=int,default=40);args=parser.parse_args();root=args.source.resolve()
 folders=sorted(p.parent for p in root.glob('*/camera.csv'));assert 1<args.keep<=len(folders)
 grid=np.linspace(0,1,101);curves=[];records=[]
 for folder in folders:
  raw=(folder/'camera.csv').read_bytes()
  with (folder/'camera.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
  x=np.array([float(r.get('segment_frame',i)) for i,r in enumerate(rows)]);y=np.array([float(r['finger_width']) if r['finger_width'] else np.nan for r in rows]);assert len(rows)>1 and np.isfinite(y).all() and np.isfinite(x).all() and np.all(np.diff(x)>0),folder
  curves.append(np.interp(grid,(x-x[0])/(x[-1]-x[0]),y));records.append(dict(trial=folder.name,frames=len(rows),camera_csv_sha256=hashlib.sha256(raw).hexdigest()))
 matrix=np.array(curves);reference=(matrix.sum(axis=0)-matrix)/(len(matrix)-1);scores=np.sqrt(np.mean((matrix-reference)**2,axis=1));order=sorted(range(len(folders)),key=lambda i:(scores[i],folders[i].name));selected=set(order[:args.keep])
 audit=root/'backup'/f'aperture_rmse_selection_{datetime.now():%Y%m%d_%H%M%S}';audit.mkdir(parents=True,exist_ok=False);reserve=audit/'excluded';reserve.mkdir()
 ranked=[]
 for rank,i in enumerate(order,1):
  dest=folders[i] if i in selected else reserve/folders[i].name
  ranked.append(dict(rank=rank,**records[i],aperture_rmse_px=float(scores[i]),action='keep' if i in selected else 'move_to_backup',original_path=str(folders[i]),destination=str(dest)))
 with (audit/'ranking.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(ranked[0]));w.writeheader();w.writerows(ranked)
 method=dict(source=str(root),original_trials=len(folders),keep=args.keep,moved=len(folders)-args.keep,signal='finger_width (aperture), unchanged px values',alignment='Full current segment, frame axis normalized to 0-1, linearly interpolated to 101 equally spaced progress points',score='RMSE against mean of all OTHER original trials; fixed cohort, computed once before any move; no iterative rescoring',ties='Trial folder name ascending',note='Curve-consistency selection, not proof of measurement error. Data moved intact; originals recoverable from excluded folder. This selection can narrow variability; report filtering and evaluate on a separately defined test set.')
 (audit/'method.json').write_text(json.dumps(method,indent=2),encoding='utf-8')
 fig,ax=plt.subplots(figsize=(10,5))
 for flag,label,color in [(False,'Moved to backup','#aaaaaa'),(True,f'Retained {args.keep}','#3485b8')]:
  first=True
  for i in range(len(matrix)):
   if (i in selected)==flag:ax.plot(grid*100,matrix[i],color=color,alpha=.45,lw=1,label=label if first else None);first=False
 ax.plot(grid*100,matrix.mean(axis=0),color='#222222',lw=2,label='Original cohort mean');ax.set(xlabel='Normalized segment progress (%)',ylabel='Fingertip aperture (px)',xlim=(0,100));ax.grid(alpha=.2);ax.legend();fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(audit/f'selection_overlay.{ext}',dpi=200)
 plt.close(fig)
 moved=[]
 try:
  for r in ranked:
   source=Path(r['original_path']);target=Path(r['destination']);assert source.parent==root and target.resolve().is_relative_to(root)
   assert hashlib.sha256((source/'camera.csv').read_bytes()).hexdigest()==r['camera_csv_sha256']
   if r['action']=='move_to_backup':
    assert not target.exists();source.rename(target);moved.append((source,target))
   assert hashlib.sha256((target/'camera.csv').read_bytes()).hexdigest()==r['camera_csv_sha256']
 except Exception:
  for source,target in reversed(moved):
   if target.exists() and not source.exists():target.rename(source)
  raise
 assert len(list(root.glob('*/camera.csv')))==args.keep
 print(json.dumps(method,indent=2));print(f'Rank {args.keep} RMSE',scores[order[args.keep-1]]);print('Audit:',audit)
if __name__=='__main__':main()
