"""Select representative live trials by measured duration and mean model MAE."""
from pathlib import Path
import csv,json,hashlib,shutil
import numpy as np

ROOT=Path(__file__).resolve().parents[1]/'New/Result/RealTime/three_model_live_20261026_210818_947238'
OUT=ROOT/'selected_best3_by_speed'
MODELS=['Function approximation','MLP (5 frames)','LSTM (full history)']

def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def first_run(mask,start=0):
    return next((i for i in range(start,len(mask)-2) if mask[i:i+3].all()),None)

def main():
    if OUT.exists():raise ValueError(f'Output already exists: {OUT}')
    with (ROOT/'trial_metrics.csv').open(encoding='utf-8-sig') as f:metrics=list(csv.DictReader(f))
    candidates=[];unclassified=[]
    for folder in sorted(ROOT.glob('trial_*')):
        if not folder.is_dir() or not (folder/'predictions.csv').is_file():continue
        with (folder/'predictions.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
        d=np.array([float(r['hand_object_signed_distance']) for r in rows]);t=np.array([float(r['elapsed_s']) for r in rows])
        valid=np.array([r['valid']=='True' for r in rows]) & np.isfinite(d)
        first=d[:10][valid[:10]]
        baseline=float(np.median(first)) if len(first) else np.nan
        sm=np.array([np.median(d[max(0,i-1):i+2][valid[max(0,i-1):i+2]]) if valid[max(0,i-1):i+2].any() else np.nan for i in range(len(d))])
        onset=first_run((sm<baseline-8)&valid)
        arrival=first_run((sm<=15)&valid,onset or 0)
        if onset is None or arrival is None or arrival<=onset:
            unclassified.append(dict(trial=folder.name,reason='No complete sustained approach to signed distance <=15 px'));continue
        duration=float(t[arrival]-t[onset])
        speed='fast' if duration<1.5 else 'normal' if duration<3 else 'slow'
        by_model={r['model']:r for r in metrics if r['trial']==folder.name}
        if set(by_model)!=set(MODELS):raise ValueError(f'Missing or extra model metrics: {folder}')
        maes=[float(by_model[m]['mae']) for m in MODELS]
        row=dict(trial=folder.name,inferred_speed=speed,measured_approach_s=duration,onset_s=float(t[onset]),arrival_s=float(t[arrival]),
                 mean_three_model_mae_px=float(np.mean(maes)),function_mae_px=maes[0],mlp_mae_px=maes[1],lstm_mae_px=maes[2],
                 common_frames=int(by_model[MODELS[0]]['frames']),original_target_s=1)
        candidates.append(row)
    selected=[];ranked=[]
    for speed in ('slow','normal','fast'):
        group=sorted([r for r in candidates if r['inferred_speed']==speed],key=lambda r:(r['mean_three_model_mae_px'],r['trial']))
        if len(group)<3:raise ValueError(f'{speed}: fewer than three eligible trials')
        for rank,row in enumerate(group,1):
            record=dict(rank_within_speed=rank,selected=rank<=3,**row)
            ranked.append(record)
            if rank<=3:selected.append(record)
    OUT.mkdir()
    for record in selected:
        source=ROOT/record['trial'];destination=OUT/record['inferred_speed']/record['trial']
        assert destination.resolve().is_relative_to(OUT.resolve())
        shutil.copytree(source,destination,copy_function=shutil.copy2)
        original={str(p.relative_to(source)):p for p in source.rglob('*') if p.is_file()}
        copied={str(p.relative_to(destination)):p for p in destination.rglob('*') if p.is_file()}
        assert set(original)==set(copied)
        for key in original:
            assert hashlib.sha256(original[key].read_bytes()).digest()==hashlib.sha256(copied[key].read_bytes()).digest()
        (destination/'selection_info.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        print('Selected and copied:',record['inferred_speed'],record['trial'],round(record['measured_approach_s'],2),round(record['mean_three_model_mae_px'],3),flush=True)
    write_csv(OUT/'selected_trials.csv',selected);write_csv(OUT/'all_trial_ranking.csv',ranked)
    if unclassified:write_csv(OUT/'unclassified_trials.csv',unclassified)
    (OUT/'method.json').write_text(json.dumps(dict(
        grouping='User-approved grouping by measured approach duration, because all recorded target labels are 1 s. fast <1.5 s; normal 1.5..<3 s; slow >=3 s.',
        timing='Median of first 10 valid signed distances as baseline; 3-point median signal only for event finding. Onset: baseline minus 8px sustained 3 valid frames. Arrival: signed distance <=15px sustained 3 valid frames. Actual CSV timestamps; threshold proxy, not contact detection.',
        ranking='Ascending mean of three saved per-trial MAEs, all evaluated on common eligible frames over the whole recorded trial, including waiting/holding. Not ranked on cropped movement-only frames.',
        purpose='Selected examples for presentation; not a replacement for all-trial evaluation.',
        source_unchanged=True,copied_files='All original trial files copied and SHA256 verified; inferred grouping stored separately without overwriting original 1 s target metadata.'),indent=2),encoding='utf-8')
    print('DONE:',OUT,flush=True)

if __name__=='__main__':main()
