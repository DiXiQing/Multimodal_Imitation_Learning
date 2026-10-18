"""Apply the user's reviewed, reversible separation; retain measurements and provenance."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

root=Path(__file__).resolve().parents[2]/'Data/TopView/TRIAL_Color_red_Size_2cm_20261017_202152_727509/grasp_segments'
root=root.resolve();backup=(root/'backup').resolve();review=root/'quality_review'
assert backup.parent==root and root.is_relative_to(Path(__file__).resolve().parents[2])
backup.mkdir(exist_ok=True)
with (review/'screening.csv').open(encoding='utf-8-sig') as f:
    reader=csv.DictReader(f);fields=list(reader.fieldnames);rows=list(reader)
selected=[r for r in rows if r['move']=='1']
assert len(selected)==5
plan=[]
for row in selected:
    src=(root/row['trial']).resolve();dest=(backup/row['trial']).resolve()
    assert src.parent==root and dest.parent==backup and src.is_dir()!=dest.is_dir()
    plan.append((src,dest,row))
manifest=[]
for src,dest,row in plan:
    location=src if src.is_dir() else dest
    before={str(p.relative_to(location)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in location.rglob('*') if p.is_file()}
    if src.is_dir():src.rename(dest)
    after={str(p.relative_to(dest)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in dest.rglob('*') if p.is_file()}
    assert before==after
    manifest.append(dict(original=str(src),backup=str(dest),reason=row['reason'],file_hashes=after))
(backup/'move_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
(backup/'README.txt').write_text(
    '5 trials set aside at user request. Four show object-center tracking jumps; one also contains missing detections. '
    'The fifth has an atypical but not confirmed invalid curve and is stored for exploratory review only. '
    'Original files unchanged; move_manifest.json records original paths and hashes. '
    'Keep this selection transparent in any reported evaluation.\n',encoding='utf-8')
fields=fields+['decision','review_conclusion']
for row in rows:
    row['decision']='backup' if row['move']=='1' else 'retained'
    row['review_conclusion']=('Visually reviewed object-center mistracking; missing frames also preserved.'
        if 'object_center_jump' in row['reason'] else
        'Exploratory curve-shape difference only; not confirmed invalid.' if row['move']=='1' else
        'No exclusion under these screening rules; not a guarantee of perfect measurement.')
with (review/'decisions.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
fig,axs=plt.subplots(1,2,figsize=(15,6.5),sharex=True,sharey=True)
for row in rows:
    moved=row['move']=='1';folder=(backup if moved else root)/row['trial']
    with (folder/'camera.csv').open(encoding='utf-8-sig') as f:data=list(csv.DictReader(f))
    x=np.array([float(r['hand_object_distance']) if r['valid']=='1' and r['hand_object_distance'] else np.nan for r in data])
    y=np.array([float(r['finger_width']) if r['valid']=='1' and r['finger_width'] else np.nan for r in data])
    color='#dc653a' if moved else '#267cb5'
    axs[0].plot(x,y,color=color,lw=1.6 if moved else 1,alpha=.9 if moved else .25,zorder=3 if moved else 1)
    if not moved:axs[1].plot(x,y,color=color,lw=1,alpha=.4)
for ax in axs:
    ax.set(xlabel='Hand-object distance (px)',ylabel='Thumb-index aperture (px)');ax.grid(alpha=.2)
axs[0].set_title('All 65 trials | retained vs backup')
axs[0].legend(handles=[Line2D([0],[0],color='#267cb5',label='Retained: 60'),
                       Line2D([0],[0],color='#dc653a',label='Backup: 5')],loc='lower center')
axs[1].set_title('Retained subset | 60 trials')
fig.suptitle('Red 2 cm | 2026-10-17 20:21 recording',fontsize=16)
fig.text(.5,.02,'Backup: 4 tracking-problem trials + 1 exploratory curve-shape outlier. '
         'Raw values unchanged; invalid frames remain gaps.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.05,1,.94))
fig.savefig(review/'retained_backup_overlay.png',dpi=180)
fig.savefig(review/'retained_backup_overlay.pdf');plt.close(fig)
assert sum(p.is_dir() and (p/'camera.csv').exists() for p in root.iterdir())==60
assert sum(p.is_dir() and (p/'camera.csv').exists() for p in backup.iterdir())==5
print('Verified: 60 retained, 5 in backup, file hashes unchanged. Overlay and decisions saved.')
for row in selected:print(row['trial'],row['reason'])
