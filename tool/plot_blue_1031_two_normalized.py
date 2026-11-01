from pathlib import Path
import os,csv,json
import numpy as np
os.environ['MPLCONFIGDIR']=str(Path(__file__).parent/'.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1]/'Data/TopView'
O=B/'Comparisons/blue_20261031_215856_vs_220155';O.mkdir(parents=True,exist_ok=True)
fig,ax=plt.subplots(figsize=(8,6));fig.subplots_adjust(left=.12,right=.98,bottom=.14,top=.97)
manifest=[]
for tag,label,color in [('215856_066525','21:58 recording','#2474B5'),('220155_064690','22:01 recording','#E58A2B')]:
 source=B/f'TRIAL_Color_blue_Size_3cm_20261031_{tag}'/'grasp_segments';trials=[]
 for folder in sorted(source.glob('grasp_[0-9][0-9][0-9]')):
  with (folder/'camera.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
  d=np.array([float(r['index_mcp_object_distance_mm']) if r.get('index_mcp_object_distance_mm') else np.nan for r in rows]);a=np.array([float(r['finger_width_mm']) if r.get('finger_width_mm') else np.nan for r in rows])
  if not np.isfinite(d[0]) or d[0]<=0:raise ValueError(f'Invalid first-frame distance: {folder}')
  settings=json.loads((folder/'settings.json').read_text(encoding='utf-8-sig'));assert float(settings['pixel_calibration']['mm_per_px'])>0
  x=d/d[0];assert x[0]==1
  ax.plot(x,a,color=color,lw=.85,alpha=.42)
  meta=json.loads((folder/'segment.json').read_text(encoding='utf-8'));trials.append(dict(trial=folder.name,initial_distance_mm=float(d[0]),frames=len(rows),review_flags=meta['review_flags']))
 ax.plot([],[],color=color,lw=2,label=f'{label} (n={len(trials)})')
 manifest.append(dict(recording=str(source.parent),color=color,trials=trials))
ax.set_xlabel(r'Normalized distance $l_t/l_0$',fontsize=13);ax.set_ylabel('Finger aperture (mm)',fontsize=13);ax.set_xlim(left=0);ax.set_ylim(bottom=0);ax.tick_params(labelsize=11);ax.grid(alpha=.2);ax.legend(loc='upper center',fontsize=10,framealpha=.95)
for ext in ['png','svg','pdf']:fig.savefig(O/f'normalized_distance_aperture_two_recordings.{ext}',dpi=300,facecolor='white')
plt.close(fig)
(O/'overlay_manifest.json').write_text(json.dumps(dict(normalization='Each segment distance divided by its first recorded frame distance, not min/max. Aperture remains mm. No smoothing or time normalization. All active segments including review flags included; backup folders excluded.',groups=manifest),indent=2),encoding='utf-8')
print('counts',[(m['recording'],len(m['trials'])) for m in manifest]);print(O)
