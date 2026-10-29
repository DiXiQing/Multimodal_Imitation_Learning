"""Plot active red dataset grouped by source recording; raw calibrated measurements."""
import os,csv,json,re
from pathlib import Path
os.environ['MPLCONFIGDIR']=str(Path(__file__).parent/'.mplconfig')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
P=Path(__file__).resolve().parents[1]/'Data/1029_D1/20mm_red'
SPECS=[('152120','15:21:20','#2171b5'),('152802','15:28:02','#e87519'),('153934','15:39:34','#249447')]
def main():
 groups=[];mapping=[];seen=set()
 for stamp,label,color in SPECS:
  with (P/f'recording_20261029_{stamp}__import_manifest.csv').open(encoding='utf-8-sig',newline='') as f:imports=list(csv.DictReader(f))
  curves=[]
  for item in imports:
   name=item['dataset_folder'];folder=P/name
   if not folder.is_dir():
    if (P/'backup'/name).is_dir():continue
    raise ValueError(f'Mapped experiment missing: {name}')
   if name in seen:raise ValueError(f'Duplicate mapping: {name}')
   seen.add(name)
   settings=json.loads((folder/'settings.json').read_text(encoding='utf-8-sig'));scale=float(settings['pixel_calibration']['mm_per_px']);assert scale>0
   with (folder/'camera.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
   v=lambda k:np.array([float(r[k]) if r.get(k) else np.nan for r in rows])
   d=v('index_mcp_object_distance_mm');a=v('finger_width_mm');t=v('t_camera');t-=t[0]
   for mm,px in [('index_mcp_object_distance_mm','index_mcp_object_distance'),('finger_width_mm','finger_width')]:
    y=v(mm);x=v(px);good=np.isfinite(x)&np.isfinite(y);assert np.allclose(y[good],x[good]*scale)
   assert np.all(np.isfinite(d)) and np.all(np.isfinite(a))
   curves.append((d,a,t));mapping.append(dict(dataset_folder=name,recording=item['source_recording'],original_segment=item['source_segment'],color=color,group_time=label,frames=len(rows),mm_per_px=scale))
  groups.append((label,color,curves))
 active={f.name for f in P.iterdir() if f.is_dir() and re.fullmatch(r'grasp_\d+',f.name)}
 assert active==seen, f'Active/mapped mismatch: {active^seen}'
 out=P/'overlays';out.mkdir(exist_ok=True)
 plots=[('distance_aperture','Index MCP (landmark 5) to object center distance (mm)','Finger aperture (mm)'),('distance_time','Actual elapsed time (s)','Index MCP distance (mm)'),('aperture_time','Actual elapsed time (s)','Finger aperture (mm)')]
 def draw(ax,k):
  for i in range(max(len(curves) for _,_,curves in groups)):
   for label,color,curves in groups:
    if i>=len(curves):continue
    d,a,t=curves[i];x,y=[(d,a),(t,d),(t,a)][k];ax.plot(x,y,color=color,lw=.95,alpha=.45)
  ax.set_xlabel(plots[k][1]);ax.set_ylabel(plots[k][2]);ax.grid(alpha=.22)
  ax.legend(handles=[Line2D([0],[0],color=color,lw=2,label=f'{label} ({len(curves)} grasps)') for label,color,curves in groups],loc='upper left' if k==0 else 'upper right')
 for k,(name,_,__) in enumerate(plots):
  fig,ax=plt.subplots(figsize=(11,7));draw(ax,k);ax.set_title(f'20mm_red | {len(seen)} experiments | three recordings');fig.tight_layout()
  for ext in ['png','pdf','svg']:fig.savefig(out/f'{name}.{ext}',dpi=190)
  if k==0:fig.savefig(out/'distance_aperture_three_recordings.png',dpi=190)
  plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(23,6))
 for k,ax in enumerate(axs):draw(ax,k)
 fig.suptitle(f'20mm_red | {len(seen)} experiments | raw calibrated measurements | three recordings');fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(out/f'combined_overlays.{ext}',dpi=190)
 plt.close(fig)
 with (out/'color_group_mapping.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(mapping[0]));w.writeheader();w.writerows(mapping)
 (out/'manifest.json').write_text(json.dumps(dict(count=len(seen),frames=sum(x['frames'] for x in mapping),groups=[dict(time=label,color=color,experiments=len(curves)) for label,color,curves in groups],distance='index_mcp_object_distance_mm',aperture='finger_width_mm',processing='All active experiments only; excluded flagged segments remain at their original recording directories. Raw data in time order; no smoothing, interpolation, time or amplitude normalization. Per-recording settings calibration verified. Actual elapsed time from first frame.'),indent=2),encoding='utf-8')
 print('Saved:',len(seen),'experiments,',sum(x['frames'] for x in mapping),'frames;',[(label,len(curves)) for label,_,curves in groups])
if __name__=='__main__':main()
