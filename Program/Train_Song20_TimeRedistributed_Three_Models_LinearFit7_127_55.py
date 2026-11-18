"""Train Song 20 mm models on redistributed trials and evaluate on the original holdout."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import re
from scipy.interpolate import BSpline

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1] / 'Data/1115_Song/20'
TRAIN_ROOT = ROOT.parent / '20_TimeRedistributed'
BASELINE_SPLIT = ROOT / 'Result' / 'Song20_Retained_LinearFit7_Spline1e-2_127_55_20261117_113412_813957' / 'split_manifest.json'
SIZE_CM = 2
TRAIN_COUNT, TEST_COUNT = 127, 55
DISTANCE = 'index_mcp_object_distance_mm'
TARGET = 'finger_width_mm'
WINDOW, VELOCITY_WINDOW, MAX_GAP, SEED = 5, 7, .10, 20261029
SPLINE_LAMBDA = 1e-2
MLP_WIDTH = 32

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def numeric(row, key):
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return np.nan

def sequence_indices(d, y, t, frame, valid):
    usable = np.isfinite(d) & np.isfinite(y) & np.isfinite(t) & np.isfinite(frame) & valid
    result, current = [], []
    for i in range(len(d)):
        if not usable[i]:
            if current: result.append(np.array(current)); current = []
            continue
        if current:
            j = current[-1]
            if frame[i] - frame[j] != 1 or not 0 < t[i] - t[j] <= MAX_GAP:
                result.append(np.array(current)); current = []
        current.append(i)
    if current: result.append(np.array(current))
    return result


VELOCITY_METHOD = "LinearFit7"


def estimate_velocity(distance, times, sequences):
    """Fit the slope of distance against actual time over seven trailing frames."""
    result = np.full(len(distance), np.nan)
    for indices in sequences:
        if len(indices) < VELOCITY_WINDOW:
            continue
        t = np.lib.stride_tricks.sliding_window_view(times[indices], VELOCITY_WINDOW)
        d = np.lib.stride_tricks.sliding_window_view(distance[indices], VELOCITY_WINDOW)
        relative_t = t - t[:, -1:]
        centered_t = relative_t - relative_t.mean(axis=1, keepdims=True)
        relative_d = d - d[:, -1:]
        result[indices[VELOCITY_WINDOW - 1:]] = (
            (centered_t * relative_d).sum(axis=1) / (centered_t ** 2).sum(axis=1)
        )
    return result


def plot_training(output):
    import matplotlib.pyplot as plt

    with (output / 'training_history.csv').open(encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    fig, ax = plt.subplots(figsize=(6, 4))
    for model in ('MLP 5f', 'LSTM Full'):
        selected = [r for r in rows if r['model'] == model]
        ax.plot([int(r['epoch']) for r in selected],
                [float(r['train_mae_mm']) for r in selected], label=model)
    ax.set(xlabel='Epoch', ylabel='Training MAE (mm)')
    ax.legend()
    ax.grid(alpha=.18)
    fig.tight_layout()
    fig.savefig(output / 'training_curves.png', dpi=180)
    plt.close(fig)


def load_trial(folder, metadata_folder=None):
    metadata_folder = folder if metadata_folder is None else metadata_folder
    required = [folder / 'camera.csv', metadata_folder / 'settings.json',
                metadata_folder / 'segment.json']
    if not all(p.is_file() for p in required):
        raise ValueError(f'{folder}: required camera.csv/settings.json/segment.json missing')
    settings = json.loads(required[1].read_text(encoding='utf-8-sig'))
    scale = (settings.get('pixel_calibration') or {}).get('mm_per_px')
    try: scale = float(scale)
    except (TypeError, ValueError): scale = np.nan
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f'{folder}: missing/invalid millimetre calibration; px fallback forbidden')
    with required[0].open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f); fields = reader.fieldnames or []; rows = list(reader)
    for key in (DISTANCE, TARGET, 't_camera', 'frame', 'valid', 'index_mcp_distance_valid'):
        if key not in fields:
            raise ValueError(f'{folder}: missing required column {key}; no fallback')
    if not rows: raise ValueError(f'{folder}: empty CSV')
    v = lambda key: np.array([numeric(r, key) for r in rows], dtype=np.float64)
    d, y, t, frame = map(v, (DISTANCE, TARGET, 't_camera', 'frame'))
    if not np.isfinite(d).any() or not np.isfinite(y).any():
        raise ValueError(f'{folder}: no finite millimetre input/target')
    if not np.isfinite(t[0]): raise ValueError(f'{folder}: missing initial timestamp')
    if not np.all(np.isfinite(frame)) or np.any(frame != np.floor(frame)) or len(np.unique(frame)) != len(frame):
        raise ValueError(f'{folder}: missing, duplicate or noninteger frame IDs')
    # Cross-check calibration when raw companion columns exist; never use them as model input.
    # Aperture targets are read directly from finger_width_mm (may be manually corrected).
    # Validate distance geometry/calibration only; never recalculate or overwrite targets.
    for mm, px in [(DISTANCE, 'index_mcp_object_distance')]:
        if px in fields:
            raw, measured = v(px), v(mm); good = np.isfinite(raw) & np.isfinite(measured)
            if not np.allclose(measured[good], raw[good] * float(scale), rtol=1e-5, atol=1e-4):
                raise ValueError(f'{folder}: {mm} inconsistent with settings calibration')
    xy_fields = ('index_mcp_x', 'index_mcp_y', 'object_x', 'object_y')
    if all(k in fields for k in xy_fields):
        expected = np.hypot(v('index_mcp_x')-v('object_x'), v('index_mcp_y')-v('object_y')) * scale
        good = np.isfinite(expected) & np.isfinite(d)
        if not np.allclose(d[good], expected[good], rtol=1e-5, atol=1e-4):
            raise ValueError(f'{folder}: millimetre distance is not landmark-5 Euclidean distance')
    if 'mm_per_px' in fields:
        row_scale = v('mm_per_px')
        if np.any(~np.isfinite(row_scale)) or not np.allclose(row_scale, float(scale)):
            raise ValueError(f'{folder}: missing/inconsistent row calibration')
    valid = np.array([r['valid'] == '1' and r['index_mcp_distance_valid'] == '1' for r in rows])
    # Teacher's physical normalization: only the initial measurement is required.
    if not valid[0] or not np.isfinite(d[0]) or d[0] <= 0:
        raise ValueError(f'{folder}: invalid first-frame distance; cannot define l0; no later-frame fallback')
    initial_distance_mm = float(d[0])
    distance_mm = d.copy()
    d = d / initial_distance_mm
    assert d[0] == 1.0
    seqs = sequence_indices(d, y, t, frame, valid)
    velocity_mm_s = estimate_velocity(distance_mm, t, seqs)
    # Distance history includes frames before the first valid velocity.
    # Every target uses only its current velocity, so there is no second warm-up.
    windows, indices, velocity_sequences = [], [], []
    for idx in seqs:
        if len(idx) < VELOCITY_WINDOW:
            continue
        velocity_sequences.append(idx[VELOCITY_WINDOW - 1:])
        for k in range(VELOCITY_WINDOW - 1, len(idx)):
            windows.append(d[idx[k-WINDOW+1:k+1]])
            indices.append(idx[k])
    seqs = velocity_sequences
    if not indices: raise ValueError(f'{folder}: no valid seven-frame common targets')
    return dict(name=folder.name, folder=folder, rows=rows, fields=fields, settings=settings,
                d=d, d_mm=distance_mm, initial_distance_mm=initial_distance_mm, y=y, t=t-t[0], unix=t, frame=frame.astype(np.int64), seqs=seqs,
                velocity_mm_s=velocity_mm_s, windows=np.array(windows), common=np.array(indices), csv_sha256=file_hash(required[0]))

def scores(y,p):
    e=p-y;den=float(np.sum((y-y.mean())**2))
    return dict(mae_mm=float(np.mean(abs(e))),rmse_mm=float(np.sqrt(np.mean(e**2))),
                r2=float(1-np.sum(e**2)/den) if den>1e-12 else None)


MC_SAMPLES = 128


def partition():
    baseline = json.loads(BASELINE_SPLIT.read_text(encoding='utf-8-sig'))
    train_ids = [entry['trial'] for entry in baseline['Train']]
    test_ids = [entry['trial'] for entry in baseline['Test']]
    if (len(train_ids), len(test_ids), len(set(train_ids + test_ids))) != (
            TRAIN_COUNT, TEST_COUNT, TRAIN_COUNT + TEST_COUNT):
        raise ValueError('Baseline split must contain 127 training and 55 distinct test trials')
    if any(not re.fullmatch(r'[0-9]{8}_[0-9]{6}', trial) for trial in train_ids + test_ids):
        raise ValueError('Unexpected trial ID in baseline split')

    train = [dict(load_trial(TRAIN_ROOT / trial, metadata_folder=ROOT / trial),
                  size_cm=SIZE_CM) for trial in train_ids]
    test = [dict(load_trial(ROOT / trial), size_cm=SIZE_CM) for trial in test_ids]
    for trial, entry in zip(test, baseline['Test']):
        trial['baseline_csv_sha256'] = entry['csv_sha256']
        if trial['csv_sha256'] != entry['csv_sha256']:
            print(f"Warning: original test CSV changed since baseline: {trial['name']}", flush=True)
    return train, test


@torch.inference_mode()
def predict_mlp_mc(model, x):
    dropout = model.net[2]
    dropout.train()
    try:
        repeated = model(x.unsqueeze(0).expand(MC_SAMPLES,-1,-1)).reshape(MC_SAMPLES,len(x))
        return torch.quantile(repeated,.5,dim=0)
    finally:
        dropout.eval()


class JointMLP(nn.Module):
    def __init__(self, width=MLP_WIDTH):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(7, width), nn.ReLU(), nn.Dropout(.1),
                                 nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, x):
        return self.net(x).reshape(-1)


class JointLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(3, 32, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(.1), nn.Linear(32, 1))

    def forward(self, features):
        hidden, _ = self.lstm(features)
        return self.head(hidden).squeeze(-1)


def features(trial, idx, device):
    columns = [trial['d'][idx], np.full(len(idx), trial['size_cm']), trial['velocity_scaled'][idx]]
    return torch.as_tensor(np.column_stack(columns), dtype=torch.float32, device=device).unsqueeze(0)


def extend_lstm_training(train, scaling):
    rng = random.Random(SEED + 1)
    augmented, records = [], []
    for trial in train:
        seconds = rng.choice((0, .5, 1, 2, 3))
        first = trial['seqs'][0]
        n = 0
        if seconds and first[0] == VELOCITY_WINDOW - 1:
            dt = float(np.median(np.diff(trial['t'][:VELOCITY_WINDOW])))
            n = round(seconds / dt)
        if n:
            times = np.r_[-dt*np.arange(n, 0, -1), trial['t']]
            distance = np.r_[np.full(n, trial['d_mm'][0]), trial['d_mm']]
            runs = [np.arange(first[-1] + n + 1)]
            runs.extend(np.arange(run[0] - VELOCITY_WINDOW + 1, run[-1] + 1) + n
                        for run in trial['seqs'][1:])
            velocity = estimate_velocity(distance, times, runs)
            trial = dict(trial, d=distance / trial['initial_distance_mm'],
                         y=np.r_[np.full(n, trial['y'][0]), trial['y']],
                         seqs=[run[VELOCITY_WINDOW - 1:] for run in runs],
                         velocity_scaled=(velocity - scaling['mean_mm_s']) / scaling['std_mm_s'])
        augmented.append(trial)
        records.append(dict(trial=trial['name'], size_cm=trial['size_cm'],
                            requested_seconds=seconds, added_frames=n))
    return augmented, records


def batches(g, label, device):
    if label == 'MLP 5f':
        idx = g['common']
        columns = [g['windows'], np.full(len(idx), g['size_cm']), g['velocity_scaled'][idx]]
        yield idx, torch.as_tensor(np.column_stack(columns), dtype=torch.float32, device=device)
    else:
        for idx in g['seqs']:
            yield idx, features(g, idx, device)


KNOTS = np.r_[np.zeros(4),np.linspace(0,1,8)[1:-1],np.ones(4)]


def design(g,low,span):
    x = (g['d'][g['common']]-low)/span
    basis = BSpline.design_matrix(x,KNOTS,3,extrapolate=True).toarray()
    blocks = [basis, (g['size_cm'] - 3) * basis]
    blocks.append(g['velocity_scaled'][g['common'], None] * basis)
    return np.column_stack(blocks)


def fit_function(train, spline_lambda):
    x = np.concatenate([g['d'][g['common']] for g in train])
    low,span = float(x.min()),float(np.ptp(x))
    if span <= 0:
        raise ValueError('No variation in training distances')
    weights = [len(g['common'])**-.5 for g in train]
    penalty = np.kron(np.eye(3),np.diff(np.eye(10),n=2,axis=0))
    a = np.vstack([*(w*design(g,low,span) for w,g in zip(weights,train)),
                   np.sqrt(spline_lambda*len(train))*penalty])
    b = np.r_[np.concatenate([w*g['y'][g['common']] for w,g in zip(weights,train)]),
              np.zeros(len(penalty))]
    return low,span,np.linalg.lstsq(a,b,rcond=None)[0]


def fit(model,label,train,mean,std,epochs,device):
    optimizer = torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    rng = random.Random(SEED)
    history = []
    for epoch in range(1,epochs+1):
        model.train()
        order = list(train)
        rng.shuffle(order)
        total = 0.
        for g in order:
            optimizer.zero_grad(set_to_none=True)
            errors = []
            for idx,x in batches(g,label,device):
                y = torch.as_tensor((g['y'][idx]-mean)/std,dtype=torch.float32,device=device)
                errors.append((model(x).reshape(-1)-y).abs())
            loss = torch.cat(errors).mean()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())*std
        history.append(dict(model=label,epoch=epoch,train_mae_mm=total/len(train)))
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print(f'{label} epoch {epoch}/{epochs}: {total/len(train):.3f} mm',flush=True)
    model.eval()
    return history


def plot_summary(output, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels = ['Function', 'MLP 5f', 'LSTM Full']
    colors = ['#3485b8', '#e59a32', '#519e58']
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (metric, title) in zip(axes, [('mae_mm', 'MAE (mm)'), ('rmse_mm', 'RMSE (mm)'), ('r2', 'R²')]):
        rows = [r for r in summary if r['size_cm'] == 'all' and r['metric'] == metric]
        means = [r['mean'] if r['mean'] is not None else np.nan for r in rows]
        errors = [r['sd'] or 0. for r in rows]
        ax.bar(labels, means, yerr=errors, capsize=4, color=colors)
        for i, (mean, sd) in enumerate(zip(means, errors)):
            if np.isfinite(mean):
                fmt = '.3f' if metric == 'r2' else '.2f'
                ax.annotate(f'{mean:{fmt}} ± {sd:{fmt}}', (i, mean+sd),
                            xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9)
        ax.set_title(title)
        ax.margins(y=.20)
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=.18)
    fig.tight_layout()
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(output / f'test_model_comparison.{ext}', dpi=180)
    plt.close(fig)


def run_holdout(train,test,args,out):
    torch.manual_seed(SEED+1)
    models = {'MLP 5f': JointMLP(), 'LSTM Full': JointLSTM()}
    low,span,coef = fit_function(train, SPLINE_LAMBDA)
    write_json(out/'function.json',dict(degree=3,knots=KNOTS.tolist(),coefficients=coef.tolist(),
               train_x_min=low,train_x_range=span,size_basis='size_cm - 3',lambda_fixed=SPLINE_LAMBDA, velocity_method=VELOCITY_METHOD, velocity_scaling=args.velocity_scaling,
               coefficient_blocks=['distance', 'distance_x_centered_size', 'distance_x_scaled_velocity']))
    y = np.concatenate([g['y'][g['common']] for g in train])
    mean,std = float(y.mean()),max(float(y.std()),1e-6)
    history = []
    for label,model in models.items():
        model.to(args.device)
        model_train = args.lstm_train if label == 'LSTM Full' else train
        history.extend(fit(model,label,model_train,mean,std,args.epochs,args.device))
        checkpoint = dict(state_dict=model.state_dict(),mean_mm=mean,std_mm=std,epochs=args.epochs,mlp_input_dim=7,velocity_method=VELOCITY_METHOD, velocity_scaling=args.velocity_scaling)
        if label == 'MLP 5f':
            checkpoint['mlp_hidden_width'] = MLP_WIDTH
        torch.save(checkpoint, out/('mlp5.pth' if label=='MLP 5f' else 'lstm_full.pth'))
    write_csv(out/'training_history.csv',history)
    metrics = []
    torch.manual_seed(SEED+101)
    with torch.inference_mode():
        for g in test:
            idx = g['common']
            pred = {label:np.full(len(g['d']),np.nan) for label in ('Function',*models)}
            pred['Function'][idx] = design(g,low,span)@coef
            for label,model in models.items():
                for ids,x in batches(g,label,args.device):
                    values = predict_mlp_mc(model,x) if label=='MLP 5f' else model(x).reshape(-1)
                    pred[label][ids] = values.cpu().numpy()*std+mean
            for label,values in pred.items():
                metrics.append(dict(size_cm=g['size_cm'],trial=g['name'],model=label, common_frames=len(idx),
                                    **scores(g['y'][idx],values[idx])))
            common = set(idx)
            write_csv(out/'Test'/f"{g['size_cm']}cm"/f"{g['name']}.csv",[
                dict(frame=int(g['frame'][i]),time_s=float(g['t'][i]),size_cm=g['size_cm'],
                     distance_over_initial=float(g['d'][i]),velocity_mm_s=float(g['velocity_mm_s'][i]),actual_mm=float(g['y'][i]),common_valid=int(i in common),
                     function_mm=float(pred['Function'][i]),mlp_5f_mm=float(pred['MLP 5f'][i]),
                     lstm_full_mm=float(pred['LSTM Full'][i])) for i in range(len(g['d']))])
    write_csv(out/'test_per_trial_metrics.csv',metrics)
    return metrics


def plot_trials(output):
    import matplotlib.pyplot as plt

    fig,ax = plt.subplots(figsize=(6,4))
    for path in sorted(output.glob('Test/*cm/*.csv')):
        with path.open(encoding='utf-8-sig',newline='') as file:
            rows = list(csv.DictReader(file))
        ax.clear()
        t = [float(r['time_s']) for r in rows]
        for key,label,color in [
            ('actual_mm','Measured','black'),
            ('function_mm','Function','#3485b8'),
            ('mlp_5f_mm','MLP 5f (MC median)','#e59a32'),
            ('lstm_full_mm','LSTM Full','#519e58')]:
            y = np.array([float(r[key]) for r in rows])
            if key != 'actual_mm':
                y[[r['common_valid'] != '1' for r in rows]] = np.nan
            ax.plot(t,y,label=label,color=color)
        ax.set(xlabel='Time (s)',ylabel='Finger aperture (mm)',title=f'{path.parent.name} | Test | {path.stem}')
        ax.legend(fontsize=8)
        ax.grid(alpha=.18)
        fig.tight_layout()
        fig.savefig(path.with_suffix('.png'),dpi=150)
    plt.close(fig)


def summarize(metrics):
    summary = []
    for size in [SIZE_CM, 'all']:
        for model in ('Function', 'MLP 5f', 'LSTM Full'):
            rows = [r for r in metrics if r['model'] == model and (size == 'all' or r['size_cm'] == size)]
            for metric in ('mae_mm', 'rmse_mm', 'r2'):
                values = [r[metric] for r in rows if r[metric] is not None]
                summary.append(dict(size_cm=size, model=model, metric=metric, n=len(values),
                                    mean=float(np.mean(values)) if values else None,
                                    sd=float(np.std(values, ddof=1)) if len(values) > 1 else None))
    return summary



def configure_velocity(train, test, args):
    values = np.concatenate([g['velocity_mm_s'][np.concatenate(g['seqs'])] for g in train])
    mean = float(values.mean())
    std = max(float(values.std()), 1e-6)
    args.velocity_scaling = dict(mean_mm_s=mean, std_mm_s=std)
    for trial in train + test:
        trial['velocity_scaled'] = (trial['velocity_mm_s'] - mean) / std


def check_data(train, test, lstm_train):
    mlp = JointMLP().eval()
    lstm = JointLSTM().eval()
    with torch.inference_mode():
        for g in train + test:
            size = SIZE_CM
            idx, x = next(batches(g, 'MLP 5f', 'cpu'))
            assert x.shape == (len(idx), 7) and torch.isfinite(x).all()
            assert torch.all(x[:, 5] == size)
            np.testing.assert_allclose(x[:, 6].numpy(), g['velocity_scaled'][idx], atol=1e-6)
            np.testing.assert_allclose(g['windows'], g['d'][idx[:, None] + np.arange(1-WINDOW, 1)])
            np.testing.assert_array_equal(np.flatnonzero(np.isfinite(g['velocity_mm_s'])), idx)
            run, sequence = next(batches(g, 'LSTM Full', 'cpu'))
            cutoff = int(run[len(run) // 2]) + 1
            prefix = np.arange(int(run[0]) - (VELOCITY_WINDOW - 1), cutoff)
            prefix_velocity = estimate_velocity(g['d_mm'][:cutoff], g['unix'][:cutoff], [prefix])
            np.testing.assert_allclose(prefix_velocity[prefix], g['velocity_mm_s'][prefix], equal_nan=True)
            assert torch.isfinite(mlp(x)).all()
            assert sequence.shape[-1] == 3 and torch.isfinite(lstm(sequence)).all()
            assert np.isfinite(design(g, 0., 1.)).all()
            np.testing.assert_array_equal(np.concatenate(g['seqs']), idx)
        for g in lstm_train:
            for ids, sequence in batches(g, 'LSTM Full', 'cpu'):
                assert torch.isfinite(sequence).all() and np.isfinite(g['y'][ids]).all()
    print(f'Common frames: train={sum(len(g["common"]) for g in train)}, '
          f'test={sum(len(g["common"]) for g in test)}', flush=True)
    print('PASS: trial split, causal 7-frame velocity, and all three model inputs', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-data', action='store_true')
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('epochs must be positive')
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    print('Loading redistributed training trials and original holdout CSVs...', flush=True)
    try:
        train, test = partition()
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    configure_velocity(train, test, args)
    args.lstm_train, static_records = extend_lstm_training(train, args.velocity_scaling)
    print(f'{len(train)} train / {len(test)} test trials (20 mm, redistributed train / original test)', flush=True)
    print(f"LSTM static prefixes: {sum(r['added_frames'] > 0 for r in static_records)} training trials", flush=True)
    print('Velocity: Linear fit using the last 7 frames | Function spline lambda: 0.01', flush=True)
    if args.check_data:
        check_data(train, test, args.lstm_train)
        return

    out = TRAIN_ROOT/'Result'/f'Song20_TimeRedistributed_LinearFit7_Spline1e-2_127_55_{datetime.now():%Y%m%d_%H%M%S_%f}'
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'split_manifest.json', dict(
        train_source=str(TRAIN_ROOT), test_source=str(ROOT), baseline_split=str(BASELINE_SPLIT),
        cube_size_mm=20, seed=SEED, epochs=args.epochs,
        changed_test_csvs=[g['name'] for g in test if g['csv_sha256'] != g['baseline_csv_sha256']],
        features=['l/l0', 'size_cm', 'scaled_velocity_mm_s'],
        distance_column=DISTANCE, aperture_column=TARGET,
        velocity_method=VELOCITY_METHOD, velocity_window_frames=VELOCITY_WINDOW,
        velocity_units='mm/s', max_future_frames=0, max_frame_gap_s=MAX_GAP,
        speed_estimator='least-squares slope of distance in mm against the last seven timestamps in seconds',
        mlp_hidden_width=MLP_WIDTH, spline_lambda=SPLINE_LAMBDA, mlp_mc_samples=MC_SAMPLES,
        mlp_layout=['x_t-4', 'x_t-3', 'x_t-2', 'x_t-1', 'x_t', 'size_cm', 'current_velocity_scaled'],
        common_support='targets from seventh through last frame of each contiguous valid run',
        lstm_history='starts at first valid velocity and resets for each trial and tracking gap',
        **{label: [dict(trial=g['name'], size_cm=g['size_cm'], csv_sha256=g['csv_sha256']) for g in group]
           for label, group in [('Train', train), ('Test', test)]}))
    write_csv(out/'split_manifest.csv', [
        dict(split=label, trial=g['name'], size_mm=20, common_frames=len(g['common']),
             csv_sha256=g['csv_sha256'])
        for label, group in [('Train', train), ('Test', test)] for g in group])
    write_json(out/'velocity_scaling.json', args.velocity_scaling)
    write_csv(out/'lstm_static_prefixes.csv', static_records)
    metrics = run_holdout(train, test, args, out)
    summary = summarize(metrics)
    write_csv(out/'test_mean_sd.csv', summary)
    plot_summary(out, summary)
    plot_training(out)
    plot_trials(out)
    print(f'Saved {out}', flush=True)


if __name__ == '__main__':
    main()
