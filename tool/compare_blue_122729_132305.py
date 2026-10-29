"""Overlay raw index-MCP distance/aperture curves using per-recording calibration."""
import os,csv,json
from pathlib import Path
os.environ['MPLCONFIGDIR']=str(Path(__file__).parent/'.mplconfig')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
BASE=Path(__file__).resolve().parents[1]/'Data/TopView'
GROUPS=[('TRIAL_Color_blue_Size_3cm_20261029_122729_581083','Previous 12:27:29','#2171b5'),('TRIAL_Color_blue_Size_3cm_20261029_132305_270675','New 13:23:05','#e66b18')]
OUT=BASE/GROUPS[1][0]/'grasp_segments'/'comparison_with_122729'

def main():
 OUT.mkdir(exist_ok=True);manifest=[];curves=[]
 for folder,label,color in GROUPS:
  source=BASE/folder;settings=json.loads((source/'settings.json').read_text(encoding='utf-8-sig'));scale=settings['pixel_calibration']['mm_per_px'];entries=json.loads((source/'grasp_segments'/'summary.json').read_text());group=[]
  for e in entries:
   f=source/'grasp_segments'/f"grasp_{e['segment']:03d}"/'camera.csv'
   with f.open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
   x=np.array([float(r['index_mcp_object_distance_mm']) if r['index_mcp_object_distance_mm'] else np.nan for r in rows]);y=np.array([float(r['finger_width_mm']) if r['finger_width_mm'] else np.nan for r in rows])
   for r in rows:
    if r['index_mcp_object_distance_mm']:assert np.isclose(float(r['index_mcp_object_distance_mm']),float(r['index_mcp_object_distance'])*scale)
    if r['finger_width_mm']:assert np.isclose(float(r['finger_width_mm']),float(r['finger_width'])*scale)
   group.append((x,y));
  curves.append((label,color,group));manifest.append(dict(source=str(source),label=label,color=color,segments=len(group),mm_per_px=scale,flagged_segments=[e['segment'] for e in entries if e['review_flags']]))
 fig,ax=plt.subplots(figsize=(11,8))
 # Alternate groups during drawing to reduce order-dependent occlusion.
 for i in range(max(len(g) for _,_,g in curves)):
  for label,color,group in curves:
   if i<len(group):ax.plot(*group[i],color=color,lw=1,alpha=.38)
 ax.set_xlabel('Index MCP (landmark 5) to object center distance (mm)');ax.set_ylabel('Finger aperture (mm)');ax.set_title('Blue 3 cm | distance-aperture curves | 2026-10-29');ax.grid(alpha=.22)
 ax.legend(handles=[Line2D([0],[0],color=c,lw=2,label=f'{label} ({len(g)} grasps)') for label,c,g in curves],loc='upper left');fig.tight_layout()
 for suffix in ['png','pdf','svg']:fig.savefig(OUT/f'distance_aperture_two_recordings.{suffix}',dpi=200)
 plt.close(fig)
 manifest=dict(groups=manifest,distance='index_mcp_object_distance_mm',aperture='finger_width_mm',processing='All individual raw segment curves; no smoothing, interpolation, averaging, normalization or rescaling between groups. Each uses its own settings calibration. All exported candidates included.')
 (OUT/'plot_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
