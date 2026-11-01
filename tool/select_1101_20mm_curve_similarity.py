from pathlib import Path
import csv, json, hashlib, shutil
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(r'C:\MineApp\Code\Multimodal_Imitation_Learning\Data\1101_D\20mm').resolve()
OUT=ROOT/'similarity_selection'
BACKUP=ROOT/'backup'/'lowest_similarity_19'
trials=sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith('grasp_') and p.name[6:].isdigit())
assert len(trials)==68, len(trials)
assert not BACKUP.exists() and not OUT.exists(), 'Selection output exists; refuse to repeat moves'
records=[]
for p in trials:
    with (p/'camera.csv').open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    x=np.array([float(r['index_mcp_object_distance_mm']) for r in rows])
    y=np.array([float(r['finger_width_mm']) for r in rows])
    assert np.isfinite(x).all() and np.isfinite(y).all() and x[0]>0,p
    assert all(float(r['mm_per_px'])>0 for r in rows),p
    records.append(dict(name=p.name,path=p,x=x/x[0],y=y,initial_distance_mm=float(x[0]),frames=len(rows)))
# One common aperture scale balances the two axes for geometric comparison only.
# Individual aperture amplitudes are preserved; no time axis is used.
y_scale=float(np.median([np.ptp(r['y']) for r in records]))
assert y_scale>0

def curve_points(r):
    pts=np.column_stack((r['x'],r['y']/y_scale))
    s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(pts,axis=0),axis=1))]
    keep=np.r_[True,np.diff(s)>1e-12]
    s,pts=s[keep],pts[keep]
    assert s[-1]>0
    q=np.linspace(0,s[-1],256)
    return np.column_stack([np.interp(q,s,pts[:,j]) for j in range(2)])
curves=[curve_points(r) for r in records]
matrix=np.zeros((len(records),len(records)))
for i in range(len(records)):
    for j in range(i):
        d=cdist(curves[i],curves[j])
        matrix[i,j]=matrix[j,i]=np.sqrt((np.mean(np.min(d,axis=0)**2)+np.mean(np.min(d,axis=1)**2))/2)
scores=np.array([np.median(np.delete(matrix[i],i)) for i in range(len(records))])
order=sorted(range(len(records)),key=lambda i:(-scores[i],records[i]['name']))
removed=set(order[:19]); retained=[r for i,r in enumerate(records) if i not in removed]
OUT.mkdir(); BACKUP.mkdir(parents=True)
manifest=[]
# Hash every trial file before moving. Moving keeps CSV, videos, frames and metadata intact.
for i,r in enumerate(records):
    files={str(p.relative_to(r['path'])):hashlib.sha256(p.read_bytes()).hexdigest() for p in r['path'].rglob('*') if p.is_file()}
    dest=BACKUP/r['name'] if i in removed else r['path']
    assert ROOT in dest.resolve().parents
    manifest.append(dict(trial=r['name'],removed=i in removed,destination=str(dest),sha256=files))
with (OUT/'file_integrity_manifest.json').open('w',encoding='utf-8') as f:json.dump(manifest,f,indent=2)
for i in order[:19]:
    p=records[i]['path']; dest=BACKUP/p.name
    assert not dest.exists()
    shutil.move(str(p),str(dest))
for m in manifest:
    dest=Path(m['destination'])
    assert {str(p.relative_to(dest)) for p in dest.rglob('*') if p.is_file()}==set(m['sha256'])
    for rel,digest in m['sha256'].items():assert hashlib.sha256((dest/rel).read_bytes()).hexdigest()==digest
with (OUT/'similarity_ranking.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['rank_most_different_first','trial','median_curve_distance_lower_is_more_similar','status','initial_distance_mm','frames'])
    for rank,i in enumerate(order,1):
        r=records[i];w.writerow([rank,r['name'],scores[i],'backup' if i in removed else 'retained',r['initial_distance_mm'],r['frames']])
method=dict(original_trial_count=68,removed_count=19,retained_count=49,excluded_existing_folder='review_segments',normalization='index_mcp_object_distance_mm / first recorded index_mcp_object_distance_mm of each trial',aperture='finger_width_mm; original values unchanged',comparison='256 equal arc-length samples for scoring only; symmetric nearest-curve-point RMS distance; rank by median distance to all other 67 trials, descending, computed once before removal',common_aperture_scale_mm=y_scale,time_normalization=False,measurement_smoothing=False,plot='All original recorded valid samples connected in chronological order, no resampling or smoothing',interpretation='Geometric similarity selection only; does not establish measurement or grasp validity',removed=[records[i]['name'] for i in order[:19]],retained=[r['name'] for r in retained],all_file_hashes_verified=True)
(OUT/'selection_method.json').write_text(json.dumps(method,indent=2),encoding='utf-8')

def plot(rs,name):
    fig,ax=plt.subplots(figsize=(5,4.8),dpi=200)
    for r in rs:ax.plot(r['x'],r['y'],color='#287db5',alpha=.42,lw=.85)
    ax.set_xlabel(r'Normalized distance $l_t/l_0$')
    ax.set_ylabel('Finger aperture (mm)')
    ax.grid(alpha=.16);ax.set_xlim(left=0);ax.set_ylim(bottom=0)
    fig.tight_layout()
    for ext in ('png','svg','pdf'):fig.savefig(OUT/(name+'.'+ext),bbox_inches='tight')
    plt.close(fig)
plot(retained,'retained_49_normalized_distance_aperture')
plot([records[i] for i in order[:19]],'removed_19_normalized_distance_aperture')
assert len(list(ROOT.glob('grasp_*')))==49
assert len(list(BACKUP.glob('grasp_*')))==19
print(json.dumps(method,indent=2))
