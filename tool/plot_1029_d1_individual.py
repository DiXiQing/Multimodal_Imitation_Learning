"""Generate a raw distance-aperture plot inside each dataset grasp folder."""
import os,csv,json,hashlib
from pathlib import Path
os.environ['MPLCONFIGDIR']=str(Path(__file__).parent/'.mplconfig')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1]/'Data/1029_D1'
def main():
 folders=sorted(f for f in P.glob('grasp_*') if f.is_dir());report=[]
 for folder in folders:
  source=folder/'camera.csv';before=hashlib.sha256(source.read_bytes()).hexdigest()
  with source.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
  settings=json.loads((folder/'settings.json').read_text(encoding='utf-8-sig'));scale=settings.get('pixel_calibration',{}).get('mm_per_px');calibrated=scale is not None and float(scale)>0
  unit='mm' if calibrated else 'px';dc='index_mcp_object_distance_mm' if calibrated else 'index_mcp_object_distance';ac='finger_width_mm' if calibrated else 'finger_width'
  def values(k):return np.array([float(r[k]) if r.get(k) else np.nan for r in rows])
  d=values(dc);a=values(ac);good=np.isfinite(d)&np.isfinite(a)
  fig,ax=plt.subplots(figsize=(8,6));ax.plot(d,a,color='#2171b5',lw=1.4,marker='.',markersize=3,label='Raw measurements (time order)')
  valid=np.flatnonzero(good)
  if len(valid):
   i,j=valid[0],valid[-1];ax.scatter(d[i],a[i],s=65,color='#2b9348',zorder=4,label='Start');ax.scatter(d[j],a[j],s=65,color='#d14936',marker='s',zorder=4,label='End')
  ax.set_xlabel(f'Index MCP (landmark 5) to object center distance ({unit})');ax.set_ylabel(f'Finger aperture ({unit})');ax.set_title(f'{folder.name} | distance-aperture');ax.grid(alpha=.25);ax.legend();fig.tight_layout();target=folder/'distance_aperture.png';fig.savefig(target,dpi=180);plt.close(fig)
  assert target.is_file() and target.stat().st_size>1000 and hashlib.sha256(source.read_bytes()).hexdigest()==before
  report.append(dict(folder=folder.name,rows=len(rows),valid_points=int(good.sum()),distance_column=dc,aperture_column=ac,unit=unit,mm_per_px=scale))
 (P/'individual_plots_manifest.json').write_text(json.dumps(dict(plots=report,processing='Raw values, original row/time order; no smoothing, interpolation or normalization; missing samples break lines; green=start, red=end.'),indent=2),encoding='utf-8')
 print(f'Generated and checked {len(report)} individual plots.',flush=True)
if __name__=='__main__':main()
