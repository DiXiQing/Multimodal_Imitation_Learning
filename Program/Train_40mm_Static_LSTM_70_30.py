"""40 mm full-history LSTM trained with synthetic static prefixes."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1] / 'Data/1101_D'
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
    windows, indices = [], []
    for idx in seqs:
        for k in range(WINDOW - 1, len(idx)):
            windows.append(d[idx[k-WINDOW+1:k+1]]); indices.append(idx[k])
    if not indices: raise ValueError(f'{folder}: no valid five-frame common targets')
    return dict(name=folder.name, folder=folder, rows=rows, fields=fields, settings=settings,
                d=d, d_mm=distance_mm, initial_distance_mm=initial_distance_mm, y=y, t=t-t[0], unix=t, frame=frame.astype(np.int64), seqs=seqs,
                windows=np.array(windows), common=np.array(indices), csv_sha256=file_hash(required[0]))

def scores(y,p):
    e=p-y;den=float(np.sum((y-y.mean())**2))
    return dict(mae_mm=float(np.mean(abs(e))),rmse_mm=float(np.sqrt(np.mean(e**2))),
                r2=float(1-np.sum(e**2)/den) if den>1e-12 else None)


class JointLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(2, 32, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(.1), nn.Linear(32, 1))

    def forward(self, features):
        hidden, _ = self.lstm(features)
        return self.head(hidden).squeeze(-1)


def features(trial, idx, device):
    x = np.column_stack((trial['d'][idx], np.full(len(idx), trial['size_cm'])))
    return torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)


def run_fold(train, test, epochs, device, output):
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    model = JointLSTM().to(device)
    targets = np.concatenate([g['y'][g['common']] for g in train])
    mean, std = float(targets.mean()), max(float(targets.std()), 1e-6)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.
        order = list(train)
        rng.shuffle(order)
        for trial in order:
            optimizer.zero_grad(set_to_none=True)
            errors = []
            for idx in trial['seqs']:
                y = torch.as_tensor((trial['y'][idx] - mean) / std, dtype=torch.float32, device=device)
                errors.append((model(features(trial, idx, device))[0] - y).abs())
            loss = torch.cat(errors).mean()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * std
        history.append(dict(epoch=epoch, train_mae_mm=total / len(train)))
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print(f'Epoch {epoch}/{epochs}: train MAE {total / len(train):.3f} mm', flush=True)
    write_csv(output / 'training_history.csv', history)
    torch.save(dict(state_dict=model.state_dict(), input_features=['l/l0', 'size_cm'],
                    mean_mm=mean, std_mm=std, epochs=epochs), output / 'lstm_full.pth')
    model.eval()
    metrics = []
    with torch.inference_mode():
        for trial in test:
            pred = np.full(len(trial['d']), np.nan)
            for idx in trial['seqs']:
                pred[idx] = model(features(trial, idx, device))[0].cpu().numpy() * std + mean
            idx = trial['common']
            size, name = trial['size_cm'], trial['name']
            hold = idx[idx < trial['hold_frames']]
            metrics.append(dict(size_cm=size, trial=name, hold_s=trial['hold_s'],
                                **scores(trial['y'][idx], pred[idx]),
                                hold_mae_mm=float(np.abs(pred[hold]-trial['y'][hold]).mean()),
                                hold_bias_mm=float((pred[hold]-trial['y'][hold]).mean()),
                                hold_prediction_range_mm=float(np.ptp(pred[hold]))))
            common = set(idx)
            write_csv(output / f'{size}cm' / f'{name}.csv', [
                dict(synthetic_initial_hold=int(i<trial['hold_frames']), frame=int(trial['frame'][i]), time_s=float(trial['t'][i]),
                     distance_mm=float(trial['d_mm'][i]), distance_over_initial=float(trial['d'][i]),
                     size_cm=size, actual_mm=float(trial['y'][i]), predicted_mm=float(pred[i]),
                     common_valid=int(i in common))
                for i in range(len(pred))])
    return metrics


def plot_results(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    def read(path):
        with path.open(encoding='utf-8-sig', newline='') as file:
            return list(csv.DictReader(file))

    summary = {(r['hold_s'],r['metric']):r for r in read(output/'test_mean_sd.csv')}
    trials = read(output/'test_per_trial_metrics.csv')
    panels = [('mae_mm','MAE (mm)'), ('rmse_mm','RMSE (mm)'), ('r2','R²'),
              ('hold_mae_mm','Static MAE (mm)'), ('hold_bias_mm','Static bias (mm)'),
              ('hold_prediction_range_mm','Static peak-to-peak (mm)')]
    groups = ['2','5','10','all']
    fig, axes = plt.subplots(2,3,figsize=(12,7))
    for ax,(metric,title) in zip(axes.flat,panels):
        rows = [summary[g,metric] for g in groups]
        means = [float(r['mean']) if r['mean'] else np.nan for r in rows]
        errors = [float(r['sd'] or 0) for r in rows]
        ax.bar(['2 s','5 s','10 s','All'],means,yerr=errors,capsize=4,color='#519e58')
        for i,(mean,sd) in enumerate(zip(means,errors)):
            if np.isfinite(mean):
                digits = 3 if metric=='r2' else 2
                ax.annotate(f'{mean:.{digits}f} ± {sd:.{digits}f}',(i,mean+sd),
                            xytext=(0,5),textcoords='offset points',ha='center',fontsize=8)
        ax.set_title(title)
        ax.margins(y=.25)
        ax.grid(axis='y',alpha=.18)
        ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ('png','svg','pdf'):
        fig.savefig(output/f'test_model_comparison.{ext}',dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7,4))
    for trial in trials:
        path = output/f"{trial['size_cm']}cm"/f"{trial['trial']}.csv"
        rows = read(path)
        t = [float(r['time_s']) for r in rows]
        hold = float(trial['hold_s'])
        ax.clear()
        ax.axvspan(0,hold,color='#e59a32',alpha=.16,label='Added static hold')
        ax.plot(t,[float(r['actual_mm']) for r in rows],color='black',label='Target')
        ax.plot(t,[float(r['predicted_mm']) for r in rows],color='#519e58',label='LSTM Full')
        ax.set(xlabel='Time (s)',ylabel='d (mm)',xlim=(0,t[-1]),
               title=f"40 mm | Test | {trial['trial']}")
        ax.grid(alpha=.18)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path.with_suffix('.png'),dpi=150)
    plt.close(fig)
    print(f'Exported summary and {len(trials)} trial plots')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-data', action='store_true')
    parser.add_argument('--plot-existing', type=Path, help='Render plots from a finished result directory')
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.plot_existing is not None:
        plot_results(args.plot_existing)
        return
    if args.epochs < 1:
        parser.error('epochs must be positive')
    torch.set_num_threads(min(4,torch.get_num_threads()))
    source = ROOT/'40mm_Static_2s_5s_10s'
    manifest = json.loads((source/'augmentation_manifest.json').read_text(encoding='utf-8-sig'))
    groups = {2:[], 5:[], 10:[]}
    seen = set()
    for record in manifest['trials']:
        g = load_trial(source/record['trial'])
        n = record['synthetic_frames']
        if record['source_csv_sha256'] in seen:
            raise ValueError('Duplicate original experiment')
        seen.add(record['source_csv_sha256'])
        if not (len(g['seqs'][0])>n and np.all(g['d'][:n+1]==1)
                and np.all(g['y'][:n+1]==g['y'][0])):
            raise ValueError(f"Invalid static prefix: {g['name']}")
        groups[record['hold_s']].append(dict(g,size_cm=4,hold_s=record['hold_s'],hold_frames=n))
    train,test = [],[]
    for seconds, group in groups.items():
        random.Random(SEED+seconds).shuffle(group)
        cut = len(group)-20
        train.extend(group[:cut])
        test.extend(group[cut:])
        print(f'{seconds}s: {cut} train / 20 test')
    assert len(train)==140 and len(test)==60
    if args.check_data:
        idx = test[0]['seqs'][0]
        model = JointLSTM().eval()
        with torch.inference_mode():
            x = features(test[0],idx,'cpu')
            pred = model(x)
            assert pred.shape==(1,len(idx)) and torch.isfinite(pred).all()
            torch.testing.assert_close(pred[:,:10],model(x[:,:10]))
        print('PASS: 200 trials, static prefixes, stratified split and causal forward pass')
        return
    out = ROOT/'Result'/f'40mm_Static_LSTM_70_30_{datetime.now():%Y%m%d_%H%M%S}'
    out.mkdir()
    write_json(out/'split_manifest.json',dict(seed=SEED,epochs=args.epochs,source=str(source),
        splits={label:[dict(trial=g['name'],hold_s=g['hold_s'],csv_sha256=g['csv_sha256']) for g in group]
                for label,group in [('Train',train),('Test',test)]}))
    rows = run_fold(train,test,args.epochs,args.device,out)
    write_csv(out/'test_per_trial_metrics.csv',rows)
    summary = []
    for seconds in (2,5,10,'all'):
        group = [r for r in rows if seconds=='all' or r['hold_s']==seconds]
        for metric in ('mae_mm','rmse_mm','r2','hold_mae_mm','hold_bias_mm','hold_prediction_range_mm'):
            values = [r[metric] for r in group if r[metric] is not None]
            summary.append(dict(hold_s=seconds,metric=metric,n=len(values),
                mean=float(np.mean(values)) if values else None,
                sd=float(np.std(values,ddof=1)) if len(values)>1 else None))
    write_csv(out/'test_mean_sd.csv',summary)
    plot_results(out)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
