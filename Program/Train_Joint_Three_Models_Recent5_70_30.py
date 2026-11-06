"""Standalone three-model 70/30 comparison with Recent5 velocity in mm/s."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import re
from scipy.interpolate import BSpline
from scipy.signal import savgol_filter

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1] / 'Data/1101_D'
SIZES = (2, 3, 4)
DISTANCE = 'index_mcp_object_distance_mm'
TARGET = 'finger_width_mm'
WINDOW, MAX_GAP, SEED = 5, .10, 20261029

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


VELOCITY_METHOD = "Recent5"


def estimate_velocity(distance, times, sequences):
    """Estimate radial velocity in mm/s, independently within each valid run."""
    result = np.full(len(distance), np.nan)
    for idx in sequences:
        if len(idx) < 9:
            continue
        measured = distance[idx]
        stamps = times[idx]
        smoothed = savgol_filter(measured, window_length=5, polyorder=2, mode='interp')
        # Shared support across all three methods avoids endpoint extrapolation.
        positions = np.arange(4, len(idx) - 3)
        if VELOCITY_METHOD == 'SmoothSG5':
            result[idx[positions]] = (
                (smoothed[positions + 1] - smoothed[positions])
                / (stamps[positions + 1] - stamps[positions])
            )
        else:
            values = smoothed if VELOCITY_METHOD == 'SmoothSG5_Recent5' else measured
            for position in positions:
                time_window = stamps[position-4:position+1]
                distance_window = values[position-4:position+1]
                centered = time_window - time_window.mean()
                result[idx[position]] = (
                    np.dot(centered, distance_window - distance_window.mean())
                    / np.dot(centered, centered)
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


def load_trial(folder):
    required = [folder / name for name in ('camera.csv', 'settings.json', 'segment.json')]
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
        if len(idx) < 9:
            continue
        velocity_sequences.append(idx[4:-3])
        for k in range(WINDOW - 1, len(idx) - 3):
            windows.append(d[idx[k-WINDOW+1:k+1]])
            indices.append(idx[k])
    seqs = velocity_sequences
    if not indices: raise ValueError(f'{folder}: no valid five-frame common targets')
    return dict(name=folder.name, folder=folder, rows=rows, fields=fields, settings=settings,
                d=d, d_mm=distance_mm, initial_distance_mm=initial_distance_mm, y=y, t=t-t[0], unix=t, frame=frame.astype(np.int64), seqs=seqs,
                velocity_mm_s=velocity_mm_s, windows=np.array(windows), common=np.array(indices), csv_sha256=file_hash(required[0]))

def scores(y,p):
    e=p-y;den=float(np.sum((y-y.mean())**2))
    return dict(mae_mm=float(np.mean(abs(e))),rmse_mm=float(np.sqrt(np.mean(e**2))),
                r2=float(1-np.sum(e**2)/den) if den>1e-12 else None)


REFERENCE = ROOT / 'Result/Joint_Combined_Function_MLP5MC_LSTMFull_20261104_133406'
MC_SAMPLES = 128


def partition():
    manifest = json.loads((REFERENCE/'split_manifest.json').read_text(encoding='utf-8-sig'))
    seen = set()
    splits = {'Train': [], 'Test': []}
    for split, records in manifest['splits'].items():
        for r in records:
            size, name = r['size_cm'], r['trial']
            if (size,name) in seen:
                raise ValueError(f'Duplicate trial: {size} cm {name}')
            seen.add((size,name))
            root = ROOT/f'{size*10}mm'
            folder = next((p for p in (root/name, root/'backup'/name)
                           if (p/'camera.csv').is_file()
                           and file_hash(p/'camera.csv')==r['csv_sha256']), None)
            if folder is None:
                raise ValueError(f'Reference CSV unavailable or changed: {size} cm {name}')
            g = dict(load_trial(folder), size_cm=size)
            splits[split].append(g)
    train, test = splits['Train'], splits['Test']
    for size in SIZES:
        if sum(g['size_cm'] == size for g in train) != 140 or sum(g['size_cm'] == size for g in test) != 60:
            raise ValueError(f'Expected 140 train and 60 test trials at {size} cm')

    def frames(trials):
        return {(int(f), float(t)) for g in trials for f, t in zip(g['frame'], g['unix']) if np.isfinite(t)}

    if frames(train) & frames(test) or {g['csv_sha256'] for g in train} & {g['csv_sha256'] for g in test}:
        raise ValueError('Train/test overlap')
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
    def __init__(self, velocity):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(7 if velocity else 6,32),nn.ReLU(),nn.Dropout(.1),
                                 nn.Linear(32,32),nn.ReLU(),nn.Linear(32,1))

    def forward(self, x):
        return self.net(x).reshape(-1)


class JointLSTM(nn.Module):
    def __init__(self, velocity):
        super().__init__()
        self.lstm = nn.LSTM(3 if velocity else 2, 32, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(.1), nn.Linear(32, 1))

    def forward(self, features):
        hidden, _ = self.lstm(features)
        return self.head(hidden).squeeze(-1)


def features(trial, idx, device):
    columns = [trial['d'][idx], np.full(len(idx), trial['size_cm'])]
    if trial['use_velocity']:
        columns.append(trial['velocity_scaled'][idx])
    return torch.as_tensor(np.column_stack(columns), dtype=torch.float32, device=device).unsqueeze(0)


def batches(g, label, device):
    if label == 'MLP 5f':
        idx = g['common']
        columns = [g['windows'], np.full(len(idx), g['size_cm'])]
        if g['use_velocity']:
            columns.append(g['velocity_scaled'][idx])
        yield idx, torch.as_tensor(np.column_stack(columns), dtype=torch.float32, device=device)
    else:
        for idx in g['seqs']:
            yield idx, features(g, idx, device)


KNOTS = np.r_[np.zeros(4),np.linspace(0,1,8)[1:-1],np.ones(4)]


def design(g,low,span):
    x = (g['d'][g['common']]-low)/span
    basis = BSpline.design_matrix(x,KNOTS,3,extrapolate=True).toarray()
    blocks = [basis, (g['size_cm'] - 3) * basis]
    if g['use_velocity']:
        blocks.append(g['velocity_scaled'][g['common'], None] * basis)
    return np.column_stack(blocks)


def fit_function(train):
    x = np.concatenate([g['d'][g['common']] for g in train])
    low,span = float(x.min()),float(np.ptp(x))
    if span <= 0:
        raise ValueError('No variation in training distances')
    weights = [len(g['common'])**-.5 for g in train]
    penalty = np.kron(np.eye(3 if train[0]['use_velocity'] else 2),np.diff(np.eye(10),n=2,axis=0))
    a = np.vstack([*(w*design(g,low,span) for w,g in zip(weights,train)),
                   np.sqrt(1e-4*len(train))*penalty])
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
    fold = 1
    torch.manual_seed(SEED+fold)
    models = {'MLP 5f':JointMLP(not args.without_velocity),'LSTM Full':JointLSTM(not args.without_velocity)}
    low,span,coef = fit_function(train)
    write_json(out/'function.json',dict(degree=3,knots=KNOTS.tolist(),coefficients=coef.tolist(),
               train_x_min=low,train_x_range=span,size_basis='size_cm - 3',lambda_fixed=1e-4, velocity_method=VELOCITY_METHOD, velocity_scaling=args.velocity_scaling,
               coefficient_blocks=['distance', 'distance_x_centered_size'] +
               ([] if args.without_velocity else ['distance_x_scaled_velocity'])))
    y = np.concatenate([g['y'][g['common']] for g in train])
    mean,std = float(y.mean()),max(float(y.std()),1e-6)
    history = []
    for label,model in models.items():
        model.to(args.device)
        history.extend(fit(model,label,train,mean,std,args.epochs,args.device))
        torch.save(dict(state_dict=model.state_dict(),mean_mm=mean,std_mm=std,epochs=args.epochs,mlp_input_dim=7 if not args.without_velocity else 6,velocity_method=VELOCITY_METHOD, velocity_scaling=args.velocity_scaling),
                   out/('mlp5.pth' if label=='MLP 5f' else 'lstm_full.pth'))
    write_csv(out/'training_history.csv',history)
    metrics = []
    torch.manual_seed(SEED+100+fold)
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
                metrics.append(dict(size_cm=g['size_cm'],trial=g['name'],model=label,
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
        ax.set(xlabel='Time (s)',ylabel='d (mm)',title=f'{path.parent.name} | Test | {path.stem}')
        ax.legend(fontsize=8)
        ax.grid(alpha=.18)
        fig.tight_layout()
        fig.savefig(path.with_suffix('.png'),dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-data',action='store_true')
    parser.add_argument('--without-velocity', action='store_true', help='Matched baseline with the same target frames')
    parser.add_argument('--epochs',type=int,default=250)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('epochs must be positive')
    torch.set_num_threads(min(4,torch.get_num_threads()))
    train, test = partition()
    velocity = not args.without_velocity
    velocity_train = np.concatenate([g['velocity_mm_s'][np.concatenate(g['seqs'])] for g in train])
    velocity_mean = float(velocity_train.mean())
    velocity_std = max(float(velocity_train.std()), 1e-6)
    args.velocity_scaling = dict(mean_mm_s=velocity_mean, std_mm_s=velocity_std)
    for g in train + test:
        g['use_velocity'] = velocity
        g['velocity_scaled'] = (g['velocity_mm_s'] - velocity_mean) / velocity_std
    print(f'{len(train)} train / {len(test)} test trials from the original 70/30 manifest', flush=True)
    if args.check_data:
        model = JointMLP(velocity).eval()
        lstm = JointLSTM(velocity).eval()
        with torch.inference_mode():
            for size in SIZES:
                g = next(g for g in test if g['size_cm'] == size)
                idx, x = next(batches(g, 'MLP 5f', 'cpu'))
                assert x.shape == (len(idx), 7 if velocity else 6)
                assert torch.all(x[:, 5] == size)
                assert np.isfinite(g['velocity_scaled'][idx]).all()
                if velocity:
                    assert np.allclose(x[:, 6].numpy(), g['velocity_scaled'][idx])
                assert np.allclose(g['windows'], g['d'][idx[:, None] + np.arange(1-WINDOW, 1)])
                assert g['common'][0] == 4
                assert torch.isfinite(predict_mlp_mc(model, x)).all()
                _, seq = next(batches(g, 'LSTM Full', 'cpu'))
                assert seq.shape[-1] == (3 if velocity else 2)
                assert torch.isfinite(lstm(seq)).all()
                assert np.isfinite(design(g, 0., 1.)).all()
        print('PASS: reference hashes, 70/30 split, velocity support and model inputs')
        return
    mode = VELOCITY_METHOD + '_CurrentV' if velocity else VELOCITY_METHOD + '_MatchedBaseline'
    out = ROOT/'Result'/f'Joint_Three_Models_{mode}_70_30_{datetime.now():%Y%m%d_%H%M%S}'
    out.mkdir(exist_ok=False)
    feature_names = ['l/l0', 'size_cm']
    if velocity:
        feature_names.append('scaled_velocity_mm_s')
    write_json(out/'split_manifest.json', dict(reference=str(REFERENCE), seed=SEED, epochs=args.epochs,
        mlp_mc_samples=MC_SAMPLES, features=feature_names, velocity_method=VELOCITY_METHOD,
        velocity_scaling=args.velocity_scaling, common_support='targets from fifth frame; last 3 frames excluded for shared support',
        mlp_layout=['x_t-4','x_t-3','x_t-2','x_t-1','x_t','size_cm','current_velocity_scaled'],
        max_future_frames=0 if VELOCITY_METHOD == 'Recent5' else (3 if VELOCITY_METHOD == 'SmoothSG5' else 2),
        splits={label:[dict(trial=g['name'], size_cm=g['size_cm'], csv_sha256=g['csv_sha256'])
                       for g in group] for label, group in [('Train', train), ('Test', test)]}))
    metrics = run_holdout(train, test, args, out)
    summary = []
    for size in [*SIZES,'all']:
        for model in ('Function','MLP 5f','LSTM Full'):
            rows = [r for r in metrics if r['model']==model and (size=='all' or r['size_cm']==size)]
            for metric in ('mae_mm','rmse_mm','r2'):
                values = [r[metric] for r in rows if r[metric] is not None]
                summary.append(dict(size_cm=size,model=model,metric=metric,n=len(values),
                    mean=float(np.mean(values)) if values else None,
                    sd=float(np.std(values,ddof=1)) if len(values)>1 else None))
    write_csv(out/'test_mean_sd.csv',summary)
    plot_summary(out, summary)
    plot_training(out)
    plot_trials(out)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
