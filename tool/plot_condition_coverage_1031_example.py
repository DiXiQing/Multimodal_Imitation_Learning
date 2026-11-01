from pathlib import Path
import csv,json,os
import numpy as np
os.environ['MPLCONFIGDIR']=str(Path(__file__).parent/'.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1];S=B/'Data/TopView/TRIAL_Color_blue_Size_3cm_20261031_220155_064690/grasp_segments';O=B/'Report/condition_coverage_20261031_220155';O.mkdir(parents=True,exist_ok=True)
entries=json.loads((S/'summary.json').read_text());rows=[];excluded=[]
for e in entries:
 if e['review_flags'] or e['closure_confirmed_row'] is None:
  excluded.append(dict(trial=f"grasp_{e['segment']:03d}",reasons=e['review_flags']));continue
 folder=S/f"grasp_{e['segment']:03d}"
 with (folder/'camera.csv').open(encoding='utf-8-sig',newline='') as f:r=list(csv.DictReader(f))
 onset=e['onset_row']-e['start_row'];closure=e['closure_confirmed_row']-e['start_row'];duration=float(r[closure]['t_camera'])-float(r[onset]['t_camera']);d=float(r[0]['index_mcp_object_distance_mm'])
 assert duration>0 and np.isfinite(d)
 rows.append(dict(trial=folder.name,grasp_duration_s=duration,initial_distance_mm=d,clip_duration_s=float(r[-1]['t_camera'])-float(r[0]['t_camera']),source=str(folder/'camera.csv')))
x=np.array([r['grasp_duration_s'] for r in rows]);y=np.array([r['initial_distance_mm'] for r in rows]);fig,ax=plt.subplots(figsize=(7.2,5.3));fig.subplots_adjust(left=.14,right=.97,bottom=.15,top=.93)
ax.scatter(x,y,s=40,color='#2474B5',alpha=.7,edgecolors='white',linewidths=.5)
ax.axvline(x.min(),color='#6A7480',ls='--',lw=.9);ax.axvline(x.max(),color='#6A7480',ls='--',lw=.9);ax.axhline(y.min(),color='#6A7480',ls='--',lw=.9);ax.axhline(y.max(),color='#6A7480',ls='--',lw=.9)
ax.set_xlabel('Grasp duration (s)',fontsize=12);ax.set_ylabel('Initial hand–object distance (mm)',fontsize=12);ax.set_title(f'Experimental condition coverage (n={len(rows)})',fontsize=13);ax.grid(alpha=.18);ax.set_axisbelow(True);ax.margins(x=.18,y=.18)
ax.text(.03,.97,f'Duration: {x.min():.2f}–{x.max():.2f} s\nInitial distance: {y.min():.1f}–{y.max():.1f} mm',transform=ax.transAxes,ha='left',va='top',fontsize=10,bbox=dict(facecolor='white',edgecolor='#CCCCCC',alpha=.92))
for ext in ['png','svg','pdf']:fig.savefig(O/f'duration_initial_distance_scatter.{ext}',dpi=300,facecolor='white')
plt.close(fig)
with (O/'per_trial_conditions.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
report=dict(included=len(rows),excluded=excluded,duration_definition='t_camera at five-frame closure confirmation minus t_camera at annotated action onset; excludes 10-frame pre-roll and 10-frame post-hold.',distance_definition='First recorded segment frame index_mcp_object_distance_mm, Euclidean landmark 5 to object centre. Unnormalized mm.',duration_min_s=float(x.min()),duration_max_s=float(x.max()),initial_distance_min_mm=float(y.min()),initial_distance_max_mm=float(y.max()))
(O/'definitions_and_ranges.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
