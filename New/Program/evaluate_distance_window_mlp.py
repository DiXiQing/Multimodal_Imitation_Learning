"""Evaluate a saved distance-history MLP on recorded grasp CSVs. Never trains."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
# Edit these two paths to run directly from an IDE.
MODEL_PATH = r'C:\MineApp\Code\Multimodal_Imitation_Learning\Data\D1\Result\distance_mlp_window5_20261022_195625_668325\models/distance_window5.pth'
TEST_FOLDER = r'C:\MineApp\Code\Multimodal_Imitation_Learning\Data\D1\Train'


class DistanceMLP(nn.Module):
    def __init__(self, inputs, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Dropout(.1),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def number(row, key):
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return np.nan


def write_csv(path, rows):
    if rows:
        with path.open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def metrics(y, p):
    valid = np.isfinite(y) & np.isfinite(p)
    y, p = y[valid], p[valid]
    if not len(y):
        return dict(frames=0, mae=None, rmse=None, r2=None)
    error = p-y
    denominator = np.sum((y-y.mean())**2)
    return dict(frames=len(y), mae=float(np.mean(abs(error))),
                rmse=float(np.sqrt(np.mean(error**2))),
                r2=float(1-np.sum(error**2)/denominator) if denominator > 1e-12 else None)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}


def discover(source):
    # A recording root with grasp_segments means evaluate its reviewed crops.
    if (source/'grasp_segments').is_dir():
        source = source/'grasp_segments'
    if (source/'camera.csv').is_file():
        return [source]
    folders = sorted(p.parent for p in source.glob('*/camera.csv'))
    if not folders:
        raise ValueError(f'No camera.csv trials found in {source}')
    return folders


def phase_for(row, meta):
    if not all(k in meta for k in ('start', 'onset', 'stable')):
        return 'unknown'
    if 'segment_frame' in row:
        position = number(row, 'segment_frame') + float(meta['start'])
    else:
        offset = float(meta.get('source_start_frame', meta['start']))-float(meta['start'])
        position = number(row, 'frame')-offset
    return 'Initial' if position < meta['onset'] else 'Grasping' if position < meta['stable'] else 'Hold'


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'), dpi=180)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=MODEL_PATH)
    parser.add_argument('--source', type=Path, default=TEST_FOLDER)
    parser.add_argument('--output', type=Path, help='New output folder; must not already exist')
    args = parser.parse_args()
    model_path = args.model
    if model_path.is_dir():
        home = model_path if list(model_path.glob('distance_window[0-9]*.pth')) else model_path/'models'
        candidates = [p for p in home.glob('distance_window[0-9]*.pth') if not p.stem.endswith('_validation')]
        if len(candidates) != 1:
            raise ValueError('Select one exact distance_windowN.pth file; model folder is empty or ambiguous.')
        model_path = candidates[0]
    saved = torch.load(model_path, map_location='cpu', weights_only=True)
    # Legacy checkpoints were trained using Euclidean distance.
    distance_column = saved.get('distance_column', 'hand_object_distance')
    if distance_column not in ('hand_object_distance', 'hand_object_signed_distance'):
        raise ValueError(f'Unsupported checkpoint distance column: {distance_column}')
    print(f'Input distance column: {distance_column}')
    window_size = saved.get('window_size')
    if saved.get('schema') != 'distance_mlp_v1' or not isinstance(window_size,int) or window_size < 2 or saved.get('input_size') != window_size:
        raise ValueError('Select a distance_mlp_v1 checkpoint with a matching history input.')
    prediction_key = f'predicted_window{window_size}_px'
    scaler = saved['scaler']
    if not all(np.isfinite(scaler[k]) for k in ('x_min', 'x_scale', 'y_mean', 'y_scale')) or min(scaler['x_scale'], scaler['y_scale']) <= 0:
        raise ValueError('Invalid checkpoint normalization.')
    gap = float(saved['max_frame_gap_s'])
    if not np.isfinite(gap) or gap <= 0:
        raise ValueError('Invalid checkpoint maximum frame gap.')
    net = DistanceMLP(window_size, saved['hidden_size'])
    net.load_state_dict(saved['state_dict'], strict=True)
    net.eval()
    folders = discover(args.source)
    out = args.output or ROOT/'New/Result'/f'offline_window{window_size}_test_{datetime.now():%Y%m%d_%H%M%S_%f}'
    out.mkdir(parents=True, exist_ok=False)
    (out/'test_predictions').mkdir()
    all_rows, per_trial, per_phase, coverage, warnings = [], [], [], [], []
    model_settings = saved.get('recording_settings', {})
    fitted = set(saved.get('fitted_trials', []))
    for folder in folders:
        with (folder/'camera.csv').open(encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            warnings.append(f'{folder.name}: empty CSV')
            continue
        if distance_column not in rows[0]:
            raise ValueError(f'{folder.name}: missing model input column {distance_column}; no automatic fallback.')
        d, y, t, frames = [np.array([number(r, k) for r in rows]) for k in
                           (distance_column, 'finger_width', 't_camera', 'frame')]
        t0 = t[np.flatnonzero(np.isfinite(t))[0]] if np.isfinite(t).any() else 0
        finite = np.isfinite(d) & np.isfinite(t) & np.isfinite(frames)
        valid = finite & np.array([number(r, 'valid') == 1 for r in rows])
        indices, windows = [], []
        for i in range(window_size-1, len(rows)):
            sl = slice(i-window_size+1, i+1)
            dt = np.diff(t[sl])
            if valid[sl].all() and np.all(np.diff(frames[sl]) == 1) and np.all(dt > 0) and np.all(dt <= gap):
                indices.append(i)
                windows.append(d[sl])
        p = np.full(len(rows), np.nan)
        if windows:
            x = (np.asarray(windows, dtype=np.float32)-scaler['x_min'])/scaler['x_scale']
            with torch.inference_mode():
                predicted = net(torch.from_numpy(x)).numpy()*scaler['y_scale']+scaler['y_mean']
            if not np.isfinite(predicted).all():
                raise ValueError(f'Nonfinite model output: {folder.name}')
            p[indices] = predicted
        meta, settings = read_json(folder/'segment.json'), read_json(folder/'settings.json')
        for key in ('object_color', 'size_cm', 'width', 'height'):
            if key in settings and key in model_settings and settings[key] != model_settings[key]:
                warnings.append(f'{folder.name}: {key} differs: model={model_settings[key]}, test={settings[key]}')
        if folder.name in fitted:
            warnings.append(f'{folder.name}: appears in checkpoint fitted_trials; not independent testing')
        trial_rows = []
        for i, row in enumerate(rows):
            phase = phase_for(row, meta)
            trial_rows.append(dict(trial=folder.name,segment_frame=row.get('segment_frame', i),
                frame=row.get('frame', ''),time_s=float(t[i]-t0) if np.isfinite(t[i]) else '',
                distance_px=float(d[i]) if np.isfinite(d[i]) else '',
                actual_aperture_px=float(y[i]) if np.isfinite(y[i]) else '',
                **{prediction_key: float(p[i]) if np.isfinite(p[i]) else ''},
                phase=phase,status='ready' if np.isfinite(p[i]) else 'invalid_detection' if not valid[i] else 'window_not_ready'))
        all_rows.extend(trial_rows)
        write_csv(out/'test_predictions'/f'{folder.name}.csv', trial_rows)
        per_trial.append(dict(trial=folder.name, **metrics(y, p)))
        for phase in ('Initial', 'Grasping', 'Hold'):
            mask = np.array([r['phase'] == phase for r in trial_rows])
            if mask.any():
                result = metrics(y[mask], p[mask])
                if result['frames']:
                    per_phase.append(dict(trial=folder.name, phase=phase, **result))
        coverage.append(dict(trial=folder.name,total_frames=len(rows),predicted_frames=int(np.isfinite(p).sum()),
            scored_frames=int((np.isfinite(p)&np.isfinite(y)).sum()),unpredicted_frames=int(np.isnan(p).sum()),
            outside_training_distance_frames=int((np.isfinite(p)&((d<scaler['x_min'])|(d>scaler['x_min']+scaler['x_scale']))).sum())))
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(t-t0, y, color='#222222', label='Measured')
        ax.plot(t-t0, p, color='#e59a32', label='Predicted')
        ax.set(title=folder.name, xlabel='Segment time (s)', ylabel='Fingertip aperture (px)')
        ax.grid(alpha=.2)
        ax.legend()
        save(fig, out/'test_predictions'/folder.name)
        print(f'{folder.name}: {per_trial[-1]}', flush=True)
    if not all_rows:
        raise ValueError('No nonempty trial CSVs.')
    actual = np.array([number(r, 'actual_aperture_px') for r in all_rows])
    predicted = np.array([number(r, prediction_key) for r in all_rows])
    overall = metrics(actual, predicted)
    if not overall['frames']:
        raise ValueError('No scorable predictions; inspect detection validity/timestamps/frame gaps.')
    for name, data in [('test_predictions', all_rows), ('test_trial_metrics', per_trial),
                       ('test_trial_phase_metrics', per_phase), ('prediction_coverage', coverage),
                       ('overall_metrics', [overall])]:
        write_csv(out/f'{name}.csv', data)
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax, key, title in zip(axes, ('mae', 'rmse', 'r2'), ('MAE (px)', 'RMSE (px)', 'R²')):
        value = overall[key]
        ax.set_title(title)
        ax.set_xticks([])
        if value is not None:
            bars = ax.bar([0], [value], width=.5, color='#e59a32')
            ax.bar_label(bars, fmt='%.3f' if key == 'r2' else '%.2f', padding=3)
            ax.margins(y=.2)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    save(fig, out/'01_overall_metrics')
    phase_summary = []
    for phase in ('Initial', 'Grasping', 'Hold'):
        part = [r for r in per_phase if r['phase'] == phase]
        if part:
            values = [r['mae'] for r in part]
            phase_summary.append(dict(phase=phase, grasps=len(part), mean_trial_mae_px=float(np.mean(values)),
                sd_trial_mae_px=float(np.std(values, ddof=1)) if len(values)>1 else 0.))
    if phase_summary:
        write_csv(out/'phase_summary.csv', phase_summary)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar([r['phase'] for r in phase_summary], [r['mean_trial_mae_px'] for r in phase_summary],
               yerr=[r['sd_trial_mae_px'] for r in phase_summary], color='#e59a32', capsize=5)
        ax.set(title='Errors by phase', ylabel='Mean per-grasp MAE (px)')
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        save(fig, out/'02_phase_errors')
    details = dict(model=str(model_path.resolve()),source=str(args.source.resolve()),trial_count=len(per_trial),distance_column=distance_column,
        overall=overall,warnings=warnings,window_size=window_size,method=f'Original CSV time/frame order; no smoothing, truncation, resampling, or training. Causal {window_size}-frame windows reset at each trial or invalid/gapped frame. Checkpoint normalization only. Overall metrics pool all finite measured/predicted pairs. Phase bars: mean per-grasp MAE with sample SD. Missing predictions stay blank.',
        model_recording_settings=model_settings)
    (out/'evaluation_info.json').write_text(json.dumps(details, indent=2), encoding='utf-8')
    for warning in warnings:
        print('NOTE:', warning)
    print('Overall:', overall)
    print('Results:', out.resolve())


if __name__ == '__main__':
    main()
