from pathlib import Path
import csv,json,os,hashlib
import numpy as np
os.environ['MPLCONFIGDIR']='tool/.mplconfig'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
base=Path(r'C:/MineApp/Code/Multimodal_Imitation_Learning')
root=base/'Data/1101_D/20mm'
new=base/'Data/TopView/TRIAL_Color_black_Size_3cm_20261101_212507_486424/grasp_segments'
manifests=[json.loads((p/'timestamp_rename_manifest.json').read_text(encoding='utf-8')) for p in (root,new)]
lookup={r['new_name']:(i,r['csv_sha256']) for i,m in enumerate(manifests) for r in m['records'] if Path(r['destination']).parent==Path(m['records'][0]['destination']).parent}
# Parent equality above selects root-level main trials; review folders are excluded.
folders=sorted(p for p in root.iterdir() if p.is_dir() and len(p.name)==15 and p.name[:8].isdigit() and p.name[8]=='_' and p.name[9:].isdigit())
assert len(folders)==102,len(folders)
colors=['#287db5','#e58b26'];labels=['19:55 recording','21:25 recording'];counts=[0,0];records=[]
fig,ax=plt.subplots(figsize=(6.4,5.2),dpi=220)
for folder in folders:
 assert folder.name in lookup,folder
 group,expected=lookup[folder.name]
 assert hashlib.sha256((folder/'camera.csv').read_bytes()).hexdigest()==expected,folder
 with (folder/'camera.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 d=np.array([float(r['index_mcp_object_distance_mm']) for r in rows]);a=np.array([float(r['finger_width_mm']) for r in rows])
 assert np.isfinite(d[0]) and d[0]>0
 ax.plot(d/d[0],a,color=colors[group],alpha=.35,lw=.85)
 counts[group]+=1
 records.append(dict(trial=folder.name,recording=labels[group],initial_distance_mm=float(d[0]),frames=len(rows)))
assert counts==[68,34],counts
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0],[0],color=c,lw=2,label=f'{label} (n={n})') for c,label,n in zip(colors,labels,counts)],fontsize=10,loc='upper center')
ax.set_xlabel(r'Normalized distance $l_t/l_0$');ax.set_ylabel('Finger aperture (mm)');ax.grid(alpha=.17);ax.set_xlim(left=0);ax.set_ylim(bottom=0)
fig.tight_layout();out=root/'overlays_two_recordings';out.mkdir(exist_ok=True)
for ext in ['png','svg','pdf']:fig.savefig(out/f'normalized_distance_aperture_two_recordings.{ext}',bbox_inches='tight')
plt.close(fig)
with (out/'trial_recording_groups.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
(out/'plot_information.json').write_text(json.dumps(dict(trials=102,groups=[dict(recording=label,color=color,trials=n) for label,color,n in zip(labels,colors,counts)],source_grouping='Matched original timestamp rename manifests and CSV SHA256; no time-based guessing',normalization='index_mcp_object_distance_mm divided by first recorded frame value per trial',aperture='finger_width_mm',smoothing=False,source_data_modified=False),indent=2),encoding='utf-8')
print('Main dataset: 102 trials; groups 19:55 = 68 (blue), 21:25 = 34 (orange); all CSV hashes verified.')
