"""Create three overlays and a trial-condition scatter plot for the 30mm blue November3 16:59 recording."""
from pathlib import Path
import csv,json,os
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).parent/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]/'Data/TopView/TRIAL_Color_blue_Size_3cm_20261103_165930_857691/grasp_segments'
entries=json.loads((ROOT/'summary.json').read_text())
trials=[]
for e in entries:
 p=ROOT/e['relative_folder']
 with (p/'camera.csv').open(encoding='utf-8-sig',newline='') as f:r=list(csv.DictReader(f))
 arr=lambda key:np.array([float(x[key]) if x.get(key) else np.nan for x in r])
 t=arr('t_camera');d=arr('index_mcp_object_distance_mm');a=arr('finger_width_mm')
 assert np.isfinite(d[0]) and d[0]>0
 trials.append(dict(meta=e,path=p,rows=r,t=t-t[0],d=d,a=a))
valid=[g for g in trials if not g['meta']['review_flags']]
plots=ROOT/'plots';plots.mkdir(exist_ok=True)
for group,out in [(valid,plots),(trials,plots/'all_candidates')]:
 out.mkdir(exist_ok=True)
 for kind,xlabel,ylabel in [('normalized_distance_aperture',r'Normalized distance $l_t/l_0$',r'$d$ (mm)'),('time_distance','Time (s)','Index MCP–object distance (mm)'),('time_aperture','Time (s)',r'$d$ (mm)')]:
  fig,ax=plt.subplots(figsize=(5.4,4.8),dpi=220)
  for g in group:
   if kind=='normalized_distance_aperture':x,y=g['d']/g['d'][0],g['a']
   elif kind=='time_distance':x,y=g['t'],g['d']
   else:x,y=g['t'],g['a']
   ax.plot(x,y,color='#287db5',alpha=.43,lw=.85)
  ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.grid(alpha=.17);ax.set_xlim(left=0);ax.set_ylim(bottom=0);fig.tight_layout()
  for ext in ['png','svg','pdf']:fig.savefig(out/f'{kind}.{ext}',bbox_inches='tight')
  plt.close(fig)
conditions=[]
for g in valid:
 e=g['meta'];onset=e['onset_row']-e['start_row'];closure=e['closure_confirmed_row']-e['start_row']
 assert e['pre_action_frames']==10 and e['post_closure_frames']==10
 conditions.append(dict(trial=g['path'].name,grasp_duration_s=float(g['t'][closure]-g['t'][onset]),initial_distance_mm=float(g['d'][0]),clip_duration_s=float(g['t'][-1]),onset_source_frame=g['rows'][onset]['frame'],closure_confirmed_source_frame=g['rows'][closure]['frame']))
x=np.array([r['grasp_duration_s'] for r in conditions]);y=np.array([r['initial_distance_mm'] for r in conditions])
fig,ax=plt.subplots(figsize=(7.2,5.3),dpi=220);fig.subplots_adjust(left=.14,right=.97,bottom=.15,top=.91)
ax.scatter(x,y,s=43,color='#287db5',alpha=.75,edgecolors='white',linewidths=.5,zorder=3)
for v in [x.min(),x.max()]:ax.axvline(v,color='#7B8590',ls='--',lw=.85)
for v in [y.min(),y.max()]:ax.axhline(v,color='#7B8590',ls='--',lw=.85)
ax.set_xlabel('trial duration(s)',fontsize=12);ax.set_ylabel('Initial hand–object distance (mm)',fontsize=12)
ax.grid(alpha=.16);ax.set_axisbelow(True)
ax.set_xlim(x.min()-.12*np.ptp(x),x.max()+.12*np.ptp(x));ax.set_ylim(y.min()-.13*np.ptp(y),y.max()+.30*np.ptp(y))
for ext in ['png','svg','pdf']:fig.savefig(plots/f'duration_initial_distance_scatter.{ext}',bbox_inches='tight')
plt.close(fig)
with (plots/'per_trial_conditions.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(conditions[0]));w.writeheader();w.writerows(conditions)
info=dict(candidates=len(entries),unflagged=len(valid),review=len(entries)-len(valid),main_plots_trials=[g['path'].name for g in valid],all_candidates_plots='all_candidates includes review segments; not a validated dataset',scatter_definition='Annotated action onset to five-frame closure confirmation, excluding pre/post margins; first recorded distance in mm',duration_range_s=[float(x.min()),float(x.max())],initial_distance_range_mm=[float(y.min()),float(y.max())],distance='Euclidean index MCP landmark 5 to object centre',distance_normalization='l/l0 only in distance-aperture plot; time-distance is raw mm',time='actual t_camera minus first timestamp; no time normalization',smoothing=False,mm_per_px=1.2,source_modified=False,total_exported_frames=sum(e['frames'] for e in entries),review_segments=[dict(trial=Path(e['relative_folder']).name,reasons=e['review_flags']) for e in entries if e['review_flags']])
(plots/'plot_information.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
for e in entries:
 v=e['validation'];assert v['csv_rows']==v['image_frames']==v['decoded_video_frames']==e['frames']
 assert (ROOT/e['relative_folder']/'distance_aperture_normalized.png').is_file()
print(json.dumps(info,indent=2))



