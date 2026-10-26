"""Raw signed-distance/aperture overlays for all D4 Train and Test trials."""
from pathlib import Path
from datetime import datetime
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]/'Data/D4'

def main():
    groups={1:[],2:[],3:[]};manifest=[];hashes={}
    for split in ('Train','Test'):
        for folder in sorted((ROOT/split).iterdir()):
            path=folder/'camera.csv'
            if not folder.is_dir() or not path.is_file():continue
            settings=json.loads((folder/'settings.json').read_text(encoding='utf-8-sig'))
            condition=int(settings['approach_guide']['target_movement_s'])
            if condition not in groups:raise ValueError(f'Unknown condition: {folder}')
            hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
            d=np.array([float(r['hand_object_signed_distance']) if r.get('hand_object_signed_distance') else np.nan for r in rows])
            a=np.array([float(r['finger_width']) if r.get('finger_width') else np.nan for r in rows])
            valid=np.array([r.get('valid')=='1' for r in rows]) & np.isfinite(d)&np.isfinite(a)
            # NaNs break lines at missing measurements; never bridge source-frame gaps.
            d[~valid]=np.nan;a[~valid]=np.nan
            frames=np.array([int(r['frame']) for r in rows])
            for i in (np.flatnonzero(np.diff(frames)!=1)+1)[::-1]:
                d=np.insert(d,i,np.nan);a=np.insert(a,i,np.nan)
            groups[condition].append((folder.name,d,a))
            manifest.append(dict(condition_s=condition,split=split,trial=folder.name,raw_frames=len(rows),valid_frames=int(valid.sum()),csv_path=str(path),csv_sha256=hashes[str(path)]))
    for condition,trials in groups.items():
        if len(trials)!=28:raise ValueError(f'{condition}s: expected 28 trials, found {len(trials)}')
    if len({r['trial'] for r in manifest})!=84 or len(set(hashes.values()))!=84:raise ValueError('Duplicate trial names/CSV files')
    out=ROOT/'Result'/f'distance_aperture_overlays_{datetime.now():%Y%m%d_%H%M%S_%f}';out.mkdir(parents=True)
    plt.rcParams.update({'font.size':12})
    all_d=np.concatenate([d for trials in groups.values() for _,d,_ in trials]);all_a=np.concatenate([a for trials in groups.values() for _,_,a in trials])
    xlim=(float(np.nanmin(all_d))-5,float(np.nanmax(all_d))+5)
    ylim=(0,float(np.nanmax(all_a))+5)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),sharex=True,sharey=True)
    for ax,(condition,trials),name in zip(axes,groups.items(),['Fast','Medium','Slow']):
        colors=plt.colormaps['turbo'](np.linspace(.03,.97,len(trials)))
        for (_,d,a),color in zip(trials,colors):ax.plot(d,a,color=color,lw=1,alpha=.75)
        ax.set(title=f'{name} ({condition} s target, n = 28)',xlabel='Signed hand–object distance (px)',xlim=xlim,ylim=ylim)
        ax.axvline(0,color='gray',ls='--',lw=.7,alpha=.7);ax.grid(alpha=.18)
    axes[0].set_ylabel('Fingertip aperture (px)')
    fig.tight_layout()
    for ext in ('png','pdf','svg'):fig.savefig(out/f'distance_aperture_by_speed.{ext}',dpi=220)
    plt.close(fig)
    for condition,trials in groups.items():
        fig,ax=plt.subplots(figsize=(6,5))
        colors=plt.colormaps['turbo'](np.linspace(.03,.97,len(trials)))
        for (_,d,a),color in zip(trials,colors):ax.plot(d,a,color=color,lw=1.1,alpha=.75)
        ax.set(xlabel='Signed hand–object distance (px)',ylabel='Fingertip aperture (px)',xlim=xlim,ylim=ylim)
        ax.axvline(0,color='gray',ls='--',lw=.7,alpha=.7);ax.grid(alpha=.18);fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(out/f'{condition}s_distance_aperture.{ext}',dpi=220)
        plt.close(fig)
    with (out/'trial_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(manifest[0]));writer.writeheader();writer.writerows(manifest)
    (out/'method.json').write_text(json.dumps(dict(source=str(ROOT),conditions={k:len(v) for k,v in groups.items()},
        cohorts='Train 21 + Test 7 per condition; all 84 trials included, no additional selection.',
        x='hand_object_signed_distance (px)',y='finger_width (px)',processing='Original chronological curves; no smoothing, time/amplitude normalization, interpolation or unit conversion. Invalid values break lines.',
        titles='1/2/3 s are target-duration labels from acquisition settings, not measured duration guarantees.',source_unchanged=True),indent=2),encoding='utf-8')
    for path,digest in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    print(out)
    print('Verified: 28 trials per speed, 84 total, original CSV hashes unchanged.')

if __name__=='__main__':main()
