"""Compare current-distance MLP with causal five-frame-distance MLP. No IMU/EMG."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from trial_paths import grasp_folders

# Edit here when running directly from an IDE. Empty SOURCE prompts for a path.
SOURCE = r'C:\MineApp\Code\Multimodal_Imitation_Learning\New\Data\SONG1\4'
TRAIN_FOLDER = 'Train'
VALIDATION_FOLDER = 'Validation'
SEED = 20261016
WINDOW_SIZE = 5                  # Fixed: [d(t-4), ..., d(t)] -> B(t)
EPOCHS = 1000                    # Maximum epochs for internal-validation stage
PATIENCE = 80
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_SIZE = 32
LOSS = 'mae'
MAX_FRAME_GAP_S = .10            # Reject a window crossing a larger timestamp gap
MODEL_SPECS = {'distance_current': 1, 'distance_window5': WINDOW_SIZE}
LABELS = {'distance_current': 'Current distance', 'distance_window5': 'Past 5 distances'}


class DistanceMLP(nn.Module):
    def __init__(self, inputs: int, hidden: int = HIDDEN_SIZE):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Dropout(.10),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def numeric(row, key):
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return np.nan


def make_windows(distance, target, timestamp, frame, valid, max_gap=MAX_FRAME_GAP_S):
    """Preserve missing rows. Both models use these exact same eligible targets."""
    windows, indices = [], []
    finite = (np.isfinite(distance) & np.isfinite(target) & np.isfinite(timestamp)
              & np.isfinite(frame) & valid)
    for end in range(WINDOW_SIZE - 1, len(distance)):
        start = end - WINDOW_SIZE + 1
        sl = slice(start, end + 1)
        dt = np.diff(timestamp[sl])
        if (finite[sl].all() and np.all(np.diff(frame[sl]) == 1)
                and np.all(dt > 0) and np.all(dt <= max_gap)):
            windows.append(distance[sl])
            indices.append(end)
    return np.asarray(windows, dtype=np.float32).reshape(-1, WINDOW_SIZE), np.asarray(indices, dtype=int)


def load_trials(source, max_gap):
    trials = []
    for folder in grasp_folders(source):
        with (folder / 'camera.csv').open(encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        d, y, t, frame = [np.array([numeric(r, key) for r in rows]) for key in
                           ('hand_object_distance', 'finger_width', 't_camera', 'frame')]
        valid = np.array([r.get('valid') == '1' for r in rows])
        windows, indices = make_windows(d, y, t, frame, valid, max_gap)
        if not len(indices):
            raise ValueError(f'{folder.name}: no valid continuous five-frame window.')
        meta = json.loads((folder / 'segment.json').read_text(encoding='utf-8-sig'))
        settings = json.loads((folder / 'settings.json').read_text(encoding='utf-8-sig'))
        phase = np.where(frame[indices] < meta['onset'], 'rest',
                         np.where(frame[indices] < meta['stable'], 'movement', 'hold'))
        trials.append(dict(name=folder.name, windows=windows, y=y[indices].astype(np.float32),
                           frame=frame[indices].astype(int), time=t[indices] - t[0],
                           window_start_frame=frame[indices - WINDOW_SIZE + 1].astype(int),
                           window_duration=t[indices] - t[indices - WINDOW_SIZE + 1],
                           phase=phase, raw_rows=len(rows), settings=settings))
    signatures = {(g['settings'].get('object_color', 'black'), g['settings']['size_cm'],
                   g['settings']['width'], g['settings']['height']) for g in trials}
    if len(signatures) > 1:
        raise ValueError('Mixed object classes or image resolutions. Train each object class separately.')
    return trials


def fit_scaler(trials):
    x = np.concatenate([g['windows'].ravel() for g in trials])
    y = np.concatenate([g['y'] for g in trials])
    scale = float(np.max(x) - np.min(x))
    if not scale > 0:
        raise ValueError('Training distances have zero range.')
    return dict(x_min=float(np.min(x)), x_scale=scale,
                y_mean=float(np.mean(y)), y_scale=max(float(np.std(y)), 1e-6))


def inputs_for(trial, inputs, scaler):
    x = trial['windows'] if inputs == WINDOW_SIZE else trial['windows'][:, -1:]
    return ((x - scaler['x_min']) / scaler['x_scale']).astype(np.float32)


def loader_for(trials, inputs, scaler, args):
    x = np.concatenate([inputs_for(g, inputs, scaler) for g in trials])
    y = np.concatenate([(g['y'] - scaler['y_mean']) / scaler['y_scale'] for g in trials])
    # Uniform sampling of windows + these weights gives each trial equal total weight.
    n = len(y)
    weights = np.concatenate([np.full(len(g['y']), n / (len(trials) * len(g['y']))) for g in trials])
    dataset = TensorDataset(torch.from_numpy(x), torch.tensor(y, dtype=torch.float32),
                            torch.tensor(weights, dtype=torch.float32))
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator)


def predict_trial(model, trial, inputs, scaler, device):
    model.eval()
    with torch.no_grad():
        value = model(torch.from_numpy(inputs_for(trial, inputs, scaler)).to(device)).cpu().numpy()
    return value * scaler['y_scale'] + scaler['y_mean']


def scores(y, prediction):
    error = np.asarray(prediction, dtype=float) - np.asarray(y, dtype=float)
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    return dict(mae=float(np.mean(abs(error))), rmse=float(np.sqrt(np.mean(error ** 2))),
                r2=float(1 - np.sum(error ** 2) / denominator) if denominator > 1e-12 else None,
                bias=float(np.mean(error)))


def validation_mae(model, trials, inputs, scaler, device):
    return float(np.mean([np.mean(abs(predict_trial(model, g, inputs, scaler, device) - g['y']))
                          for g in trials]))


def epoch_step(model, loader, optimizer, loss_name, device, y_scale):
    model.train()
    total_loss = total_absolute = count = 0
    for x, y, weight in loader:
        x, y, weight = x.to(device), y.to(device), weight.to(device)
        optimizer.zero_grad(set_to_none=True)
        error = model(x) - y
        loss = ((error.abs() if loss_name == 'mae' else error.square()) * weight).mean()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * len(y)
        total_absolute += float((error.detach().abs() * weight).sum()) * y_scale
        count += len(y)
    return total_loss / count, total_absolute / count


def checkpoint(model, name, inputs, scaler, args, epoch, trials):
    return dict(schema='distance_mlp_v1', model_name=name,
                state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                input_size=inputs, hidden_size=args.hidden, dropout=.10,
                window_size=inputs, comparison_required_context=WINDOW_SIZE,
                input_order='oldest to newest; current frame included',
                target='current-frame fingertip aperture; no historical aperture inputs',
                scaler=scaler, loss=args.loss, epochs=epoch, seed=args.seed,
                max_frame_gap_s=args.max_gap, recording_settings=trials[0]['settings'],
                fitted_trials=[g['name'] for g in trials])


def train_one(name, inputs, fit_trials, val_trials, args, output, device, log):
    seed_all(args.seed)
    scaler = fit_scaler(fit_trials)
    model = DistanceMLP(inputs, args.hidden).to(device)
    loader = loader_for(fit_trials, inputs, scaler, args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_mae, best_epoch, wait = float('inf'), 0, 0
    history = []
    log(f'\n[{name}] Select epoch using {len(fit_trials)} fit / {len(val_trials)} validation trials')
    for epoch in range(1, args.epochs + 1):
        loss, train_mae = epoch_step(model, loader, optimizer, args.loss, device, scaler['y_scale'])
        val_mae = validation_mae(model, val_trials, inputs, scaler, device)
        history.append(dict(model=name, stage='selection', epoch=epoch,
                            train_loss=loss, train_mae_px=train_mae, validation_mae_px=val_mae))
        if val_mae < best_mae - 1e-4:
            best_mae, best_epoch, wait = val_mae, epoch, 0
            torch.save(checkpoint(model, name, inputs, scaler, args, epoch, fit_trials),
                       output / 'models' / f'{name}_validation.pth')
        else:
            wait += 1
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs or wait >= args.patience:
            log(f'  epoch {epoch:4d}/{args.epochs} | train MAE={train_mae:.3f}px '
                f'| val MAE={val_mae:.3f}px | best epoch={best_epoch}')
        if wait >= args.patience:
            log(f'  Early stopping: {args.patience} epochs without validation improvement.')
            break
    # Keep the exact validation-selected checkpoint. Never fit on Validation.
    best = torch.load(output / 'models' / f'{name}_validation.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(best['state_dict'])
    model.eval()
    best['validation_trials'] = [g['name'] for g in val_trials]
    best['selection_note'] = 'Minimum validation macro-MAE checkpoint; normalization and fitting use Train only.'
    torch.save(best, output / 'models' / f'{name}.pth')
    log(f'[{name}] Saved best checkpoint at epoch {best_epoch}; validation is never added to training.')
    return model, scaler, history, dict(model=name, selected_epoch=best_epoch,
                                      best_validation_mae_px=best_mae)


def load_folder_partitions(source, max_gap):
    import hashlib
    train_dir, validation_dir = source / TRAIN_FOLDER, source / VALIDATION_FOLDER
    if not train_dir.is_dir() or not validation_dir.is_dir():
        raise ValueError(f'Required folders: {train_dir} and {validation_dir}')
    train, validation = load_trials(train_dir, max_gap), load_trials(validation_dir, max_gap)
    if not train or not validation:
        raise ValueError('Train and Validation must both contain grasp trial folders.')
    if {g['name'] for g in train} & {g['name'] for g in validation}:
        raise ValueError('Duplicate trial names across Train and Validation.')
    hashes = [{hashlib.sha256((folder / g['name'] / 'camera.csv').read_bytes()).hexdigest()
               for g in group} for folder,group in [(train_dir,train),(validation_dir,validation)]]
    if hashes[0] & hashes[1]:
        raise ValueError('Identical camera.csv data exists in both Train and Validation.')
    signatures = {(g['settings'].get('object_color','black'),g['settings']['size_cm'],
                   g['settings']['width'],g['settings']['height']) for g in train+validation}
    if len(signatures) != 1:
        raise ValueError('Train and Validation object classes/resolutions differ.')
    return train,validation


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=SOURCE)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--weight-decay', type=float, default=WEIGHT_DECAY)
    parser.add_argument('--hidden', type=int, default=HIDDEN_SIZE)
    parser.add_argument('--loss', choices=['mae', 'mse'], default=LOSS)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--max-gap', type=float, default=MAX_FRAME_GAP_S)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--log-every', type=int, default=25)
    parser.add_argument('--output', '--output-dir', type=Path)
    args = parser.parse_args()
    text = str(args.source).strip() or input('请输入数据文件夹绝对路径：\n> ').strip()
    source = Path(text.strip('\"\''))
    if not source.is_absolute() or not source.is_dir():
        parser.error('请输入存在的数据文件夹绝对路径。')
    if (args.epochs < 1 or args.patience < 1
            or args.batch_size < 1 or args.hidden < 1 or args.lr <= 0 or args.max_gap <= 0
            or args.log_every < 1 or args.weight_decay < 0):
        parser.error('Invalid counts or training parameters.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --device cpu.')
    device = torch.device('cuda' if args.device == 'cuda' or
                          (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    try:
        fit_trials, val_trials = load_folder_partitions(source, args.max_gap)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    trials = fit_trials + val_trials
    validation = val_trials
    output = args.output or source / 'Result' / f'distance_mlp_window5_{datetime.now():%Y%m%d_%H%M%S_%f}'
    if output.exists():
        parser.error(f'Output already exists: {output}')
    (output / 'models').mkdir(parents=True)
    (output / 'validation_predictions').mkdir()
    def log(message=''):
        print(message, flush=True)
        with (output / 'training_log.txt').open('a', encoding='utf-8') as f:
            f.write(message + '\n')
    started = time.perf_counter()
    log('=== Current-distance MLP vs causal five-frame MLP ===')
    log(f'Data: {source}\nDevice: {device} | Loss: {args.loss} | Seed: {args.seed}')
    log(f'Train: {len(fit_trials)} trials from {source / TRAIN_FOLDER}')
    log(f'Validation: {len(val_trials)} trials from {source / VALIDATION_FOLDER}')
    log('Validation selects the best checkpoint; these are validation metrics, not independent test metrics.')
    log('Both models predict the SAME target frames; first four frames and discontinuous windows are omitted.')
    log(f'Maximum selection epochs: {args.epochs}; early-stop patience: {args.patience}')
    log(f'Results: {output.resolve()}')
    split = dict(seed=args.seed, strategy='Fixed folders; no random trial split.',
                 train_directory=str(source / TRAIN_FOLDER),validation_directory=str(source / VALIDATION_FOLDER),
                 train_trials=[g['name'] for g in fit_trials],
                 validation_trials=[g['name'] for g in val_trials],
                 test_trials=[],selection='Minimum validation macro-MAE checkpoint; no combined refit. Train-only normalization.')
    (output / 'split.json').write_text(json.dumps(split, indent=2), encoding='utf-8')
    split_rows = []
    val_names = {g['name'] for g in val_trials}
    for g in trials:
        split_rows.append(dict(trial=g['name'], split='validation' if g['name'] in val_names else 'train',
            internal_role='validation' if g['name'] in val_names else 'fit',
            raw_rows=g['raw_rows'], eligible_target_frames=len(g['y']),
            omitted_rows=g['raw_rows'] - len(g['y'])))
    write_csv(output / 'data_split.csv', split_rows)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings.update(source=str(source), output=str(output), window_size=WINDOW_SIZE,
                    normalization='Input MinMax and output standardization, fitted only on the fitting partition at each stage.',
                    window_rule='Five consecutive valid source frames, increasing actual timestamps, no gap over max_gap.',
                    models=MODEL_SPECS)
    (output / 'model_information.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    trained, histories, selections = {}, [], []
    for name, inputs in MODEL_SPECS.items():
        model, scaler, history, selection = train_one(name, inputs, fit_trials, val_trials,
                                                      args, output, device, log)
        trained[name] = (model, scaler)
        histories.extend(history)
        selections.append(selection)
    write_csv(output / 'training_history.csv', histories)
    write_csv(output / 'epoch_selection.csv', selections)
    log('\n=== Validation evaluation: identical target frames for both models ===')
    metrics_rows, all_predictions = [], []
    pooled = {name: [] for name in MODEL_SPECS}
    truth = []
    for g in validation:
        predictions = {name: predict_trial(model, g, MODEL_SPECS[name], scaler, device)
                       for name, (model, scaler) in trained.items()}
        truth.append(g['y'])
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(g['time'], g['y'], color='#222222', lw=2, label='Measured')
        for name, color in zip(MODEL_SPECS, ['#3485b8', '#e59a32']):
            pred = predictions[name]
            pooled[name].append(pred)
            score = scores(g['y'], pred)
            metrics_rows.append(dict(model=name, trial=g['name'], frames=len(pred), **score))
            ax.plot(g['time'], pred, color=color, lw=1.6, label=LABELS[name])
            log(f'  {g["name"]} | {name}: MAE={score["mae"]:.2f}px | RMSE={score["rmse"]:.2f}px')
        trial_rows = []
        for i in range(len(g['y'])):
            row = dict(trial=g['name'], source_frame=int(g['frame'][i]),
                window_start_frame=int(g['window_start_frame'][i]), window_duration_s=float(g['window_duration'][i]),
                time_s=float(g['time'][i]), phase=str(g['phase'][i]), distance_px=float(g['windows'][i, -1]),
                actual_aperture_px=float(g['y'][i]),
                predicted_current_px=float(predictions['distance_current'][i]),
                predicted_window5_px=float(predictions['distance_window5'][i]))
            trial_rows.append(row)
        all_predictions.extend(trial_rows)
        write_csv(output / 'validation_predictions' / f'{g["name"]}.csv', trial_rows)
        ax.set(title=g['name'], xlabel='Time from segment start (s)', ylabel='Aperture (px)')
        ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
        fig.savefig(output / 'validation_predictions' / f'{g["name"]}.png', dpi=160); plt.close(fig)
    write_csv(output / 'validation_trial_metrics.csv', metrics_rows)
    write_csv(output / 'validation_predictions.csv', all_predictions)
    summary = []
    for name in MODEL_SPECS:
        score = scores(np.concatenate(truth), np.concatenate(pooled[name]))
        per_trial = [r for r in metrics_rows if r['model'] == name]
        summary.append(dict(model=name, frames=sum(r['frames'] for r in per_trial), **score,
            trial_mae_mean=float(np.mean([r['mae'] for r in per_trial])),
            trial_mae_sd=float(np.std([r['mae'] for r in per_trial], ddof=1)) if len(per_trial) > 1 else 0.,
            trial_rmse_mean=float(np.mean([r['rmse'] for r in per_trial]))))
    write_csv(output / 'overall_model_summary.csv', summary)
    phase_rows = []
    for phase in ('rest', 'movement', 'hold'):
        subset = [r for r in all_predictions if r['phase'] == phase]
        if not subset:
            continue
        actual = np.array([r['actual_aperture_px'] for r in subset])
        for name, key in [('distance_current', 'predicted_current_px'), ('distance_window5', 'predicted_window5_px')]:
            phase_rows.append(dict(model=name, phase=phase, frames=len(subset),
                **scores(actual, np.array([r[key] for r in subset]))))
    write_csv(output / 'validation_phase_metrics.csv', phase_rows)
    from plot_test_phase_errors import export_phase_errors
    export_phase_errors(output, all_predictions, prefix='validation')
    plot_results(output, summary, metrics_rows, histories, validation)
    log('\n=== Overall validation metrics (all eligible validation frames pooled) ===')
    for row in summary:
        r2 = f'{row["r2"]:.3f}' if row['r2'] is not None else 'N/A'
        log(f'{LABELS[row["model"]]}: MAE={row["mae"]:.2f}px | RMSE={row["rmse"]:.2f}px | R2={r2}')
    log(f'Total time: {time.perf_counter() - started:.1f}s')
    log(f'\nSaved results to: {output.resolve()}')
    for name in ['validation_model_comparison.png', 'validation_metrics_chart.png', 'validation_predictions/*.png',
                 'training_curves.png', 'overall_model_summary.csv', 'validation_phase_metrics.csv',
                 'validation_phase_errors.png', 'validation_trial_phase_metrics.csv', 'validation_phase_summary.csv',
                 'data_split.csv', 'models/distance_current.pth', 'models/distance_window5.pth']:
        log('  ' + name)
    (output / 'RESULT.txt').write_text((output / 'training_log.txt').read_text(encoding='utf-8'), encoding='utf-8')


def plot_results(output, summary, per_trial, histories, validation):
    colors = ['#3485b8', '#e59a32']
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, key, title in zip(axs, ['mae', 'rmse', 'r2'], ['MAE (px)', 'RMSE (px)', 'R\u00b2']):
        values = [r[key] if r[key] is not None else np.nan for r in summary]
        bars = ax.bar(['Current', 'Window 5'], values, color=colors)
        ax.bar_label(bars, fmt='%.3f' if key == 'r2' else '%.2f', padding=3)
        ax.set_title(title); ax.margins(y=.2); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('Same validation trials and target frames | current distance vs five-frame history')
    fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(output / 'validation_model_comparison.png', dpi=180)
    fig.savefig(output / 'validation_model_comparison.pdf'); plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(14, max(4.5, .45 * (len(validation) + 1) + 2)))
    pos = np.arange(len(validation) + 1)
    for ax, key, title in zip(axs, ['mae', 'rmse', 'r2'], ['MAE (px)', 'RMSE (px)', 'R\u00b2']):
        for j, (name, color) in enumerate(zip(MODEL_SPECS, colors)):
            by_trial = {r['trial']: r for r in per_trial if r['model'] == name}
            overall = next(r for r in summary if r['model'] == name)
            values = [by_trial[g['name']][key] for g in validation] + [overall[key]]
            values = [v if v is not None else np.nan for v in values]
            ax.barh(pos + (j - .5) * .34, values, height=.32, color=color, label=LABELS[name])
        ax.set_yticks(pos, [f'Validation {i:02d}' for i in range(1, len(validation) + 1)] + ['Overall'])
        ax.invert_yaxis(); ax.set_title(title); ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
    axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output / 'validation_metrics_chart.png', dpi=170)
    fig.savefig(output / 'validation_metrics_chart.pdf'); plt.close(fig)
    write_csv(output / 'validation_label_mapping.csv', [dict(label=f'Validation {i:02d}', trial=g['name']) for i, g in enumerate(validation, 1)])
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax,name in zip(axs,MODEL_SPECS):
        selection = [r for r in histories if r['model']==name]
        ax.plot([r['epoch'] for r in selection],[r['train_mae_px'] for r in selection],label='Train (dropout active)')
        ax.plot([r['epoch'] for r in selection],[r['validation_mae_px'] for r in selection],label='Validation')
        ax.set(title=LABELS[name],xlabel='Epoch',ylabel='MAE (px)');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'training_curves.png',dpi=160);plt.close(fig)



if __name__ == '__main__':
    main()
