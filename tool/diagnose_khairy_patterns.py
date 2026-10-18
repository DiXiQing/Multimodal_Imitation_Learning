from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt




ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_window_mlp as base

def main():
    source = ROOT / 'Data/Exprement Data/Khairy/3'
    output = ROOT / 'Report/khairy_pattern_diagnosis'
    output.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in source.rglob('grasp_data_smoothed.csv') if 'Result' not in p.parts)
    names = [p.parent.name for p in paths]
    assert len(names) == len(set(names))
    frames = [pd.read_csv(p) for p in paths]
    columns = ['finger_width', 'object_area', 'forward_acceleration']
    grid = np.linspace(0, 1, 101)
    curves = np.array([np.array([np.interp(grid, (d.t_camera-d.t_camera.iloc[0])/(d.t_camera.iloc[-1]-d.t_camera.iloc[0]), d[c]) for c in columns]) for d in frames])
    scales = np.std(curves, axis=(0, 2))
    distance = np.sqrt(np.mean(((curves[:, None]-curves[None, :])/scales[None, None, :, None])**2, axis=(2, 3)))
    rows = []
    for name, d in zip(names, frames):
        camera = pd.read_csv(paths[names.index(name)].parent/'camera.csv')
        peak = int(np.argmax(d.finger_width))
        baseline = np.median(camera.object_area.iloc[:min(10, peak)])
        rows.append(dict(trial=name, path=str(paths[names.index(name)].parent), frames=len(d), duration=d.t_camera.iloc[-1]-d.t_camera.iloc[0], fw_peak=d.finger_width.max(), fw_peak_progress=100*grid[np.argmax(curves[names.index(name), 0])], oa_peak_progress=100*grid[np.argmax(curves[names.index(name), 1])], acc_peak_progress=100*grid[np.argmax(curves[names.index(name), 2])], oa_dip_ratio=camera.object_area.iloc[:peak+1].min()/baseline, camera_max_gap=np.diff(d.t_camera).max(), valid=np.isfinite(d[columns]).all().all() and (d.finger_width>0).all() and (d.object_area>0).all()))
    table = pd.DataFrame(rows)
    eligible = np.flatnonzero(table.valid & (table.oa_dip_ratio >= .90) & (table.camera_max_gap < .10) & (table.frames >= 40))
    assert len(eligible) >= 15
    neighborhoods = [eligible[np.argsort(distance[i, eligible])[:15]] for i in eligible]
    scores = [distance[np.ix_(n, n)].mean() for n in neighborhoods]
    chosen = neighborhoods[int(np.argmin(scores))]
    medoid = chosen[np.argmin(distance[np.ix_(chosen, chosen)].mean(axis=1))]
    table['distance_to_medoid'] = distance[:, medoid]
    table['candidate_15'] = [i in chosen for i in range(len(names))]
    for cols, label in [(['object_area'], 'oa'), (['object_area', 'forward_acceleration'], 'oa_acc')]:
        windows = [base.make_trial_windows(d, cols, 15)[:2] for d in frames]
        errors = []
        for i, (x, y) in enumerate(windows):
            others = [j for j in range(len(names)) if j != i and table.valid.iloc[j]]
            train_x = np.concatenate([windows[j][0].reshape(len(windows[j][0]), -1) for j in others])
            train_y = np.concatenate([windows[j][1] for j in others])
            mean = train_x.mean(axis=0); scale = train_x.std(axis=0); scale[scale < 1e-8] = 1
            z = (train_x-mean)/scale; ym = train_y.mean(); weights = np.linalg.solve(z.T@z+100*np.eye(z.shape[1]), z.T@(train_y-ym))
            pred = ((x.reshape(len(x), -1)-mean)/scale)@weights+ym
            errors.append(np.sqrt(np.mean((pred-y)**2)))
        table['loo_ridge_rmse_'+label] = errors
    table.sort_values('distance_to_medoid').to_csv(output/'all_trials.csv', index=False, encoding='utf-8-sig')
    table.iloc[chosen].sort_values('distance_to_medoid').to_csv(output/'candidate_15.csv', index=False, encoding='utf-8-sig')
    target = names.index('TRIAL_Size_3cm_20261016_105203')
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for k, ax in enumerate(axes):
        for i in chosen:
            ax.plot(grid*100, curves[i, k], color='gray', alpha=.35, lw=1)
        ax.plot(grid*100, np.median(curves[chosen, k], axis=0), color='black', lw=2, label='Candidate 15 median')
        ax.plot(grid*100, curves[target, k], color='red', lw=2, label='105203')
        ax.set_ylabel(columns[k]); ax.grid(alpha=.2); ax.legend()
    axes[-1].set_xlabel('Elapsed trial time (%)')
    fig.tight_layout(); fig.savefig(output/'105203_vs_patterns.png', dpi=150); plt.close(fig)
    (output/'method.md').write_text('Exploratory cohort, not an independent test set. All 54 trials, including backup, were inspected numerically. Common elapsed-time normalization to 101 points preserves relative timing across modalities. A single pooled standard deviation per modality preserves trial amplitude differences. Candidate selection minimizes joint FW/OA/acceleration curve distance after basic validity, OA dip, frame count and camera-gap screening. No prediction errors enter selection. Raw-frame review is still required before restoring any backup trial. Leave-one-trial-out Ridge with fixed alpha=100 is a diagnostic baseline using training-fold-only scaling, not a replacement for MLP evaluation or evidence of sensor corruption. No original trials are moved or overwritten.\n', encoding='utf-8')
    print(table.iloc[target].to_string())
    print('MEDOID', names[medoid], 'ELIGIBLE', len(eligible))
    print(table.iloc[chosen][['trial','fw_peak','fw_peak_progress','oa_peak_progress','acc_peak_progress']].to_string(index=False))
    print('MEDIANS', table.iloc[chosen][['fw_peak','fw_peak_progress','oa_peak_progress','acc_peak_progress']].median().to_dict())
    print('OUTPUT', output)

if __name__ == '__main__':
    main()

