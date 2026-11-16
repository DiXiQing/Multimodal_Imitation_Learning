"""Joint full-history LSTM with inputs [l/l0, cube size in cm]."""
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
REFERENCES = {2: '20mm_20261103_185038', 3: '30mm_20261103_184459', 4: '40mm_20261103_172526'}
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


def load_folds():
    folds = [([], []) for _ in range(5)]
    baseline = {}
    for size, reference in REFERENCES.items():
        result = ROOT / 'Result' / reference
        manifest = json.loads((result / 'split_manifest.json').read_text(encoding='utf-8-sig'))
        trials = {}
        for record in manifest['records']:
            name = record['trial']
            parent = ROOT / f'{size * 10}mm'
            candidates = [parent / name, parent / 'backup' / name]
            folder = next((p for p in candidates if (p / 'camera.csv').is_file()
                           and file_hash(p / 'camera.csv') == record['csv_sha256']), None)
            if folder is None:
                raise ValueError(f'{size} cm {name}: original CSV matching reference hash unavailable')
            trial = load_trial(folder)
            trial['size_cm'] = size
            trials[name] = trial
        tested = []
        for split in manifest['folds']:
            fold = int(split['fold']) - 1
            train, test = split['train'], split['test']
            if set(train) & set(test) or set(train + test) != set(trials):
                raise ValueError(f'Invalid partition: {reference}, fold {fold + 1}')
            folds[fold][0].extend(trials[n] for n in train)
            folds[fold][1].extend(trials[n] for n in test)
            tested.extend(test)
        if sorted(tested) != sorted(trials):
            raise ValueError(f'Incomplete or duplicate tests: {reference}')
        with (result / 'oof_per_trial_metrics.csv').open(encoding='utf-8-sig') as file:
            for row in csv.DictReader(file):
                if row['split'] == 'Test' and row['model'] == 'LSTM Full':
                    baseline[size, row['trial']] = row
    for train, test in folds:
        for key in (lambda g: g['csv_sha256'],
                    lambda g: (g['size_cm'], g['name'])):
            if {key(g) for g in train} & {key(g) for g in test}:
                raise ValueError('Train/Test trial overlap')
        frames = lambda group: {(int(f), float(t)) for g in group
                                for f, t in zip(g['frame'], g['unix']) if np.isfinite(t)}
        if frames(train) & frames(test):
            raise ValueError('Train/Test recorded frame overlap')
    return folds, baseline


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


def run_fold(train, test, baseline, fold, epochs, device, output):
    torch.manual_seed(SEED + fold - 1)
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
            print(f'Fold {fold} epoch {epoch}/{epochs}: train MAE {total / len(train):.3f} mm', flush=True)
    write_csv(output / 'training_history.csv', history)
    torch.save(dict(state_dict=model.state_dict(), input_features=['l/l0', 'size_cm'],
                    output_mean_mm=mean, output_std_mm=std, epochs=epochs), output / 'joint_lstm.pth')
    model.eval()
    metrics = []
    with torch.inference_mode():
        for trial in test:
            pred = np.full(len(trial['d']), np.nan)
            for idx in trial['seqs']:
                pred[idx] = model(features(trial, idx, device))[0].cpu().numpy() * std + mean
            idx = trial['common']
            size, name = trial['size_cm'], trial['name']
            old = baseline[size, name]
            if int(old['frames']) != len(idx):
                raise ValueError(f'{name}: scoring frame count differs from reference')
            metrics.append(dict(fold=fold, size_cm=size, trial=name,
                                **scores(trial['y'][idx], pred[idx]),
                                reference_mae_mm=float(old['mae_mm']),
                                reference_rmse_mm=float(old['rmse_mm']),
                                reference_r2=float(old['r2'])))
            common = set(idx)
            write_csv(output / f'{size}cm' / f'{name}.csv', [
                dict(frame=int(trial['frame'][i]), time_s=float(trial['t'][i]),
                     distance_mm=float(trial['d_mm'][i]), distance_over_initial=float(trial['d'][i]),
                     size_cm=size, actual_mm=float(trial['y'][i]), predicted_mm=float(pred[i]),
                     common_valid=int(i in common))
                for i in range(len(pred))])
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-data', action='store_true')
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('epochs must be positive')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    folds, baseline = load_folds()
    for i, (train, test) in enumerate(folds, 1):
        counts = {s: (sum(g['size_cm'] == s for g in train), sum(g['size_cm'] == s for g in test))
                  for s in REFERENCES}
        print(f'Fold {i}: train/test counts by size cm {counts}', flush=True)
        for g in test:
            old = baseline[g['size_cm'], g['name']]
            assert int(old['fold']) == i and int(old['frames']) == len(g['common'])
    if args.check_data:
        model = JointLSTM().eval()
        with torch.inference_mode():
            for size in REFERENCES:
                trial = next(g for g in folds[0][1] if g['size_cm'] == size)
                x = features(trial, trial['seqs'][0], 'cpu')
                assert x.shape[-1] == 2 and torch.all(x[..., 1] == size)
                y = model(x)
                assert y.shape == x.shape[:2] and torch.isfinite(y).all()
                torch.testing.assert_close(model(x[:, :5]), y[:, :5])
        print('PASS: matched data, held-out folds and two-input causal forward pass.')
        return
    output = ROOT / 'Result' / f'Joint_LSTM_CV5_{datetime.now():%Y%m%d_%H%M%S}'
    output.mkdir(exist_ok=False)
    write_json(output / 'split_manifest.json', dict(references=REFERENCES, epochs=args.epochs,
        folds=[{key: [dict(size_cm=g['size_cm'], trial=g['name'], csv_sha256=g['csv_sha256'])
                      for g in group] for key, group in [('train', tr), ('test', te)]} for tr, te in folds]))
    rows = []
    for i, (train, test) in enumerate(folds, 1):
        rows.extend(run_fold(train, test, baseline, i, args.epochs, args.device, output / f'fold_{i:02d}'))
    write_csv(output / 'oof_per_trial_metrics.csv', rows)
    summary = []
    for size in [*REFERENCES, 'all']:
        group = [r for r in rows if size == 'all' or r['size_cm'] == size]
        for metric in ('mae_mm', 'rmse_mm', 'r2'):
            paired = [r for r in group if r[metric] is not None]
            new = np.array([r[metric] for r in paired])
            old = np.array([r['reference_' + metric] for r in paired])
            summary.append(dict(size_cm=size, metric=metric, n=len(paired),
                                joint_mean=new.mean(), joint_sd=new.std(ddof=1),
                                separate_mean=old.mean(), separate_sd=old.std(ddof=1),
                                mean_paired_change=(new - old).mean()))
    write_csv(output / 'joint_vs_separate.csv', summary)
    print(f'Saved {output}')


if __name__ == '__main__':
    main()
