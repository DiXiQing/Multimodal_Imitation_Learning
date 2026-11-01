from pathlib import Path
import csv,json,os
import numpy as np
os.environ['MPLCONFIGDIR']=str(Path(__file__).parent/'.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1]
S=B/'Data/1101_D/20mm'
O=S/'condition_coverage';O.mkdir(exist_ok=True)
folders=sorted(p for p in S.iterdir() if p.is_dir() and p.name.startswith('grasp_') and p.name[6:].isdigit())
assert len(folders)==49
rows=[]
for folder in folders:
 e=json.loads((folder/'segment.json').read_text(encoding='utf-8-sig'))
 with (folder/'camera.csv').open(encoding='utf-8-sig',newline='') as f:r=list(csv.DictReader(f))
 onset=e['onset_row']-e['start_row'];closure=e['closure_confirmed_row']-e['start_row']
 assert 0<=onset<=closure<len(r),folder
 duration=float(r[closure]['t_camera'])-float(r[onset]['t_camera']);d=float(r[0]['index_mcp_object_distance_mm'])
 assert duration>0 and np.isfinite(d)
 rows.append(dict(trial=folder.name,grasp_duration_s=duration,initial_distance_mm=d,clip_duration_s=float(r[-1]['t_camera'])-float(r[0]['t_camera']),action_onset_source_frame=r[onset]['frame'],closure_confirmed_source_frame=r[closure]['frame'],source=str(folder/'camera.csv')))
x=np.array([r['grasp_duration_s'] for r in rows]);y=np.array([r['initial_distance_mm'] for r in rows])
fig,ax=plt.subplots(figsize=(7.2,5.3));fig.subplots_adjust(left=.14,right=.97,bottom=.15,top=.91)
ax.scatter(x,y,s=43,color='#2474B5',alpha=.75,edgecolors='white',linewidths=.5,zorder=3)
for v in [x.min(),x.max()]:ax.axvline(v,color='#7B8590',ls='--',lw=.85)
for v in [y.min(),y.max()]:ax.axhline(v,color='#7B8590',ls='--',lw=.85)
ax.set_xlabel('Grasp duration (s)',fontsize=12);ax.set_ylabel('Initial hand–object distance (mm)',fontsize=12)
ax.set_title(f'Experimental condition coverage (n={len(rows)})',fontsize=13)
ax.grid(alpha=.16);ax.set_axisbelow(True)
ax.set_xlim(x.min()-.12*np.ptp(x),x.max()+.12*np.ptp(x))
ax.set_ylim(y.min()-.13*np.ptp(y),y.max()+.30*np.ptp(y))
ax.text(.03,.97,f'Duration: {x.min():.2f}–{x.max():.2f} s\nInitial distance: {y.min():.1f}–{y.max():.1f} mm',transform=ax.transAxes,ha='left',va='top',fontsize=10,bbox=dict(facecolor='white',edgecolor='#CCCCCC',alpha=.95))
for ext in ['png','svg','pdf']:fig.savefig(O/f'duration_initial_distance_scatter.{ext}',dpi=300,facecolor='white')
plt.close(fig)
with (O/'per_trial_conditions.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
report=dict(included=len(rows),selection='49 retained trials in dataset root; backup and review_segments excluded',duration_definition='t_camera at five-frame closure confirmation minus t_camera at annotated action onset; excludes pre-action and post-closure retained frames.',distance_definition='First recorded segment frame index_mcp_object_distance_mm; Euclidean MediaPipe landmark 5 to object centre. Unnormalized mm.',duration_min_s=float(x.min()),duration_max_s=float(x.max()),initial_distance_min_mm=float(y.min()),initial_distance_max_mm=float(y.max()),source_data_modified=False)
(O/'definitions_and_ranges.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
