"""Overlay raw distance/aperture measurements for the active 20mm_red dataset."""
import os,csv,json,re
from pathlib import Path
os.environ['MPLCONFIGDIR']=str(Path(__file__).parent/'.mplconfig')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1]/'Data/1029_D1/20mm_red'
def main():
 trials=[]
 for folder in sorted(P.iterdir()):
  if not folder.is_dir() or not re.fullmatch(r'grasp_\d+',folder.name):continue
  settings=json.loads((folder/'settings.json').read_text(encoding='utf-8-sig'));scale=(settings.get('pixel_calibration') or {}).get('mm_per_px')
  if scale is None or float(scale)<=0:raise ValueError(f'Missing mm calibration: {folder}')
  with (folder/'camera.csv').open(encoding='utf-8-sig',newline='') as f:
   reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
  for key in ['index_mcp_object_distance_mm','finger_width_mm','t_camera']:
   if key not in fields:raise ValueError(f'{folder}: missing {key}')
  v=lambda k:np.array([float(r[k]) if r.get(k) else np.nan for r in rows]);d=v('index_mcp_object_distance_mm');a=v('finger_width_mm');t=v('t_camera');t=t-t[0]
  trials.append((folder.name,d,a,t))
 if not trials:raise ValueError('No active experiments')
 out=P/'overlays';out.mkdir(exist_ok=True)
 specs=[('distance_aperture','Index MCP (landmark 5) to object center distance (mm)','Finger aperture (mm)'),('distance_time','Actual elapsed time (s)','Index MCP distance (mm)'),('aperture_time','Actual elapsed time (s)','Finger aperture (mm)')]
 def draw(ax,k):
  for n,(_,d,a,t) in enumerate(trials):
   x,y=[(d,a),(t,d),(t,a)][k];ax.plot(x,y,lw=.9,alpha=.65,color=plt.cm.turbo(n/max(1,len(trials)-1)))
  ax.set_xlabel(specs[k][1]);ax.set_ylabel(specs[k][2]);ax.grid(alpha=.22)
 for k,(name,_,__) in enumerate(specs):
  fig,ax=plt.subplots(figsize=(11,7));draw(ax,k);ax.set_title(f'20mm_red | {len(trials)} experiments | raw measurements');fig.tight_layout()
  for ext in ['png','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=190)
  plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(22,6))
 for k,ax in enumerate(axs):draw(ax,k)
 fig.suptitle(f'20mm_red | {len(trials)} experiments | raw mm measurements | actual elapsed timestamps');fig.tight_layout();fig.savefig(out/'combined_overlays.png',dpi=190);plt.close(fig)
 (out/'manifest.json').write_text(json.dumps(dict(experiments=[x[0] for x in trials],count=len(trials),frames=sum(len(x[1]) for x in trials),distance='index_mcp_object_distance_mm',aperture='finger_width_mm',processing='Raw values; no smoothing/interpolation/normalization. Time = actual t_camera minus first timestamp. Only direct grasp_* folders; backup, Train/Test and derived folders excluded.'),indent=2),encoding='utf-8')
 print('Exported',len(trials),'experiments,',sum(len(x[1]) for x in trials),'frames to',out)
if __name__=='__main__':main()
