"""Teacher-requested l/l0 overlays, alongside recorded-mm plots; no data mutation."""
import os,csv,json,re
from pathlib import Path
os.environ['MPLCONFIGDIR']=str(Path(__file__).parent/'.mplconfig')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
P=Path(__file__).resolve().parents[1]/'Data/1029_D1'
SPECS=[('20mm_red','Red 20 mm','#d94a40',20),('3cm_blue','Blue 30 mm','#2171b5',30),('40mm_black','Black 40 mm','#333333',40)]
def main():
 groups=[];audit=[]
 for folder,label,color,size in SPECS:
  curves=[]
  for trial in sorted((P/folder).iterdir()):
   if not trial.is_dir() or not re.fullmatch(r'grasp_\d+',trial.name):continue
   settings=json.loads((trial/'settings.json').read_text(encoding='utf-8-sig'));scale=float(settings['pixel_calibration']['mm_per_px']);assert np.isfinite(scale) and scale>0
   with (trial/'camera.csv').open(encoding='utf-8-sig',newline='') as f:r=list(csv.DictReader(f))
   v=lambda k:np.array([float(x[k]) if x.get(k) else np.nan for x in r])
   d=v('index_mcp_object_distance_mm');a=v('finger_width_mm');t=v('t_camera');t-=t[0]
   assert np.isfinite(d[0]) and d[0]>0 and r[0]['valid']=='1' and r[0]['index_mcp_distance_valid']=='1'
   norm=d/d[0];assert norm[0]==1.
   for mm,px in [('index_mcp_object_distance_mm','index_mcp_object_distance'),('finger_width_mm','finger_width')]:
    x=v(px);y=v(mm);good=np.isfinite(x)&np.isfinite(y);assert np.allclose(x[good]*scale,y[good])
   curves.append(dict(name=trial.name,d=d,a=a,t=t,norm=norm))
   audit.append(dict(object_directory=folder,trial=trial.name,physical_object_size_mm=size,recorded_size_cm=settings.get('size_cm'),initial_distance_mm=float(d[0]),initial_normalized_distance=float(norm[0]),duration_s=float(t[-1]),within_1_to_3_s=bool(1<=t[-1]<=3),frames=len(r),mm_per_px=scale))
  if not curves:raise ValueError(folder)
  groups.append((folder,label,color,curves))
 out=P/'normalized_distance_overlays';out.mkdir(exist_ok=True)
 def axis(ax,normalized):
  ax.set_xlabel('Index MCP distance / initial distance (l/l0)' if normalized else 'Index MCP distance (mm)');ax.set_ylabel('Finger aperture (mm)');ax.grid(alpha=.22)
  if normalized:ax.axvline(1,color='gray',ls=':',lw=.7)
 def draw(ax,curves,color,normalized):
  for g in curves:ax.plot(g['norm'] if normalized else g['d'],g['a'],color=color,alpha=.35,lw=.85)
 fig,axes=plt.subplots(3,2,figsize=(15,15))
 for row,(folder,label,color,curves) in zip(axes,groups):
  for ax,normalized in zip(row,[False,True]):draw(ax,curves,color,normalized);axis(ax,normalized);ax.set_title(f'{label} | {len(curves)} trials | '+('l/l0' if normalized else 'mm'))
  dest=P/folder/'overlays';dest.mkdir(exist_ok=True)
  for normalized,name in [(False,'distance_aperture_active_mm'),(True,'normalized_distance_aperture')]:
   f,ax=plt.subplots(figsize=(10,7));draw(ax,curves,color,normalized);axis(ax,normalized);ax.set_title(f'{label} | {len(curves)} trials | raw aperture (mm)');f.tight_layout()
   for ext in ['png','pdf']:f.savefig(dest/f'{name}.{ext}',dpi=180)
   plt.close(f)
 fig.suptitle('Same trials: recorded millimetres vs initial-distance normalization | aperture unchanged (mm)');fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(out/f'three_objects_mm_vs_normalized.{ext}',dpi=170)
 plt.close(fig)
 fig,ax=plt.subplots(figsize=(12,8))
 for _,label,color,curves in groups:draw(ax,curves,color,True)
 axis(ax,True);ax.legend(handles=[Line2D([0],[0],color=color,lw=2,label=f'{label} ({len(curves)} trials)') for _,label,color,curves in groups]);ax.set_title('Initial-distance normalization | three objects | aperture in mm');fig.tight_layout()
 for ext in ['png','pdf','svg']:fig.savefig(out/f'three_objects_normalized_overlay.{ext}',dpi=180)
 plt.close(fig)
 # Preserve three-recording colours for the red dataset comparison requested earlier.
 red={g['name']:g for folder,_,_,curves in groups if folder=='20mm_red' for g in curves}
 colors=[('152120','15:21:20','#2171b5'),('152802','15:28:02','#e87519'),('153934','15:39:34','#249447')]
 fig,ax=plt.subplots(figsize=(11,7));handles=[]
 for stamp,label,color in colors:
  with (P/'20mm_red'/f'recording_20261029_{stamp}__import_manifest.csv').open(encoding='utf-8-sig',newline='') as f:items=list(csv.DictReader(f))
  selected=[red[x['dataset_folder']] for x in items if x['dataset_folder'] in red];draw(ax,selected,color,True);handles.append(Line2D([0],[0],color=color,lw=2,label=f'{label} ({len(selected)} trials)'))
 axis(ax,True);ax.legend(handles=handles);ax.set_title(f'Red 20 mm | {len(red)} trials | l/l0 normalization');fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(P/'20mm_red'/'overlays'/f'normalized_distance_aperture_three_recordings.{ext}',dpi=180)
 plt.close(fig)
 with (out/'trial_initial_distance_audit.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(audit[0]));w.writeheader();w.writerows(audit)
 summary=[]
 for folder,label,color,curves in groups:
  rows=[r for r in audit if r['object_directory']==folder];initial=[r['initial_distance_mm'] for r in rows]
  summary.append(dict(object=label,trials=len(rows),initial_mm_min=min(initial),initial_mm_median=float(np.median(initial)),initial_mm_max=max(initial),within_1_to_3_s=sum(r['within_1_to_3_s'] for r in rows)))
 (out/'manifest.json').write_text(json.dumps(dict(formula='l_normalized(t) = index_mcp_object_distance_mm(t) / index_mcp_object_distance_mm(first recorded frame)',aperture='finger_width_mm unchanged',time='Actual timestamps unchanged',numerical_processing='No smoothing, interpolation, aperture scaling, time normalization or future-based baseline; no source CSV modified',calibration='User confirmed all scales were measured with a 30 mm reference block; original calibration retained',metadata_note='User confirmed black physical side is 40 mm although source settings label it 3 cm; physical label only, no rescaling applied',groups=summary),indent=2),encoding='utf-8')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
