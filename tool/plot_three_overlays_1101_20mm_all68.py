from pathlib import Path
import csv,json,os
import numpy as np
os.environ['MPLCONFIGDIR']='tool/.mplconfig'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(r'C:/MineApp/Code/Multimodal_Imitation_Learning/Data/1101_D/20mm')
folders=sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith('grasp_') and p.name[6:].isdigit())
assert len(folders)==68
out=root/'overlays_all68';out.mkdir(exist_ok=True)
trials=[]
for folder in folders:
 with (folder/'camera.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 values=lambda col:np.array([float(r[col]) if r.get(col) else np.nan for r in rows])
 d=values('index_mcp_object_distance_mm');a=values('finger_width_mm');t=values('t_camera')
 assert np.isfinite(d[0]) and d[0]>0 and np.isfinite(t).all() and np.all(np.diff(t)>0),folder
 trials.append((t-t[0],d,a))
plots=[('normalized_distance_aperture_68_trials',r'Normalized distance $l_t/l_0$','Finger aperture (mm)',lambda t,d,a:(d/d[0],a)),('time_distance_68_trials','Time (s)','Index MCP–object distance (mm)',lambda t,d,a:(t,d)),('time_aperture_68_trials','Time (s)','Finger aperture (mm)',lambda t,d,a:(t,a))]
for name,xlabel,ylabel,xy in plots:
 fig,ax=plt.subplots(figsize=(5.4,4.8),dpi=220)
 for t,d,a in trials:
  x,y=xy(t,d,a);ax.plot(x,y,color='#287db5',alpha=.4,lw=.85)
 ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.grid(alpha=.17);ax.set_xlim(left=0);ax.set_ylim(bottom=0);fig.tight_layout()
 for ext in ('png','svg','pdf'):fig.savefig(out/f'{name}.{ext}',bbox_inches='tight')
 plt.close(fig)
(out/'three_plots_information.json').write_text(json.dumps(dict(trials=68,trial_names=[p.name for p in folders],distance_column='index_mcp_object_distance_mm',aperture_column='finger_width_mm',normalized_distance='distance divided by first recorded distance per trial; only first graph',time='t_camera minus first t_camera per trial; seconds; not normalized',time_distance_unit='mm',time_aperture_unit='mm',smoothing=False,source_data_modified=False,excluded='backup, review_segments and derived folders'),indent=2),encoding='utf-8')
print('Saved three separate overlays for 68 trials: PNG, SVG and PDF each.')
