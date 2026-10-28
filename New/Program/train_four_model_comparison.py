"""Compare single-point function approximation, window MLP, window LSTM and full-trial causal LSTM on identical target frames. No training with --check-data."""
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
SOURCE = r'C:\MineApp\Code\Multimodal_Imitation_Learning\Data\Blue_3cm_two_positions_20261028'
TEST_FRACTION = .25             # Random split by complete trial, never by individual frame
TRAIN_FOLDER = 'Train'
TEST_FOLDER = 'Test'
SEED = 20261016
WINDOW_SIZE = 5                 # Includes current frame; change here or use --window-size
DISTANCE_COLUMN = 'hand_object_signed_distance'  # Signed projection; old models used hand_object_distance
EPOCHS = 400                     # Fixed full passes over Train; save the LAST epoch.
FUNCTION_KIND = 'smoothing_spline'
SPLINE_LAMBDA = 1e-6  # Fixed in advance for this run, not selected using Test.
POLY_DEGREE = 3
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_SIZE = 32
LOSS = 'mae'
MAX_FRAME_GAP_S = .10            # Reject a window crossing a larger timestamp gap
WINDOW_MODEL = f'distance_window{WINDOW_SIZE}'
PREDICTION_KEY = f'predicted_window{WINDOW_SIZE}_px'
LSTM_MODEL = 'distance_lstm_continuous'
LSTM_PREDICTION_KEY = 'predicted_lstm_continuous_px'
WINDOW_LSTM_MODEL = f'distance_lstm_window{WINDOW_SIZE}'
WINDOW_LSTM_KEY = f'predicted_lstm_window{WINDOW_SIZE}_px'
MODEL_SPECS = {'function_approximation': 1, WINDOW_MODEL: WINDOW_SIZE, WINDOW_LSTM_MODEL: WINDOW_SIZE, LSTM_MODEL: 0}
LABELS = {'function_approximation': 'Function approximation', WINDOW_MODEL: f'MLP ({WINDOW_SIZE} frames)', WINDOW_LSTM_MODEL: f'LSTM ({WINDOW_SIZE} frames)', LSTM_MODEL: 'LSTM (full history)'}


class DistanceMLP(nn.Module):
    def __init__(self, inputs: int, hidden: int = HIDDEN_SIZE):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Dropout(.10),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FunctionApproximation:
    """Serializable single-distance polynomial or cubic smoothing spline."""
    def __init__(self, definition):
        self.definition=definition

    def predict(self,x):
        spec=self.definition
        if spec['kind']=='polynomial':
            return np.polynomial.polynomial.polyval(x,spec['coefficients'])
        from scipy.interpolate import BSpline
        return BSpline(spec['knots'],spec['coefficients'],spec['degree'],extrapolate=True)(x)


def fit_fixed_function(fit,args,output,log):
    from scipy.interpolate import make_smoothing_spline
    scaler=fit_scaler(fit)
    x=np.concatenate([g['windows'][:,-1] for g in fit]);y=np.concatenate([g['y'] for g in fit])
    x=(x-scaler['x_min'])/scaler['x_scale'];y=(y-scaler['y_mean'])/scaler['y_scale']
    weights=np.concatenate([np.full(len(g['y']),1/len(g['y'])) for g in fit])
    if args.function_kind=='polynomial':
        design=np.polynomial.polynomial.polyvander(x,args.poly_degree)
        coef=np.linalg.lstsq(design*np.sqrt(weights)[:,None],y*np.sqrt(weights),rcond=None)[0]
        spec=dict(kind='polynomial',degree=args.poly_degree,coefficients=coef.tolist())
    else:
        unique,inverse=np.unique(x,return_inverse=True)
        if len(unique)<5:raise ValueError('Smoothing spline requires at least 5 distinct Train distances.')
        summed=np.bincount(inverse,weights=weights)
        mean_y=np.bincount(inverse,weights=weights*y)/summed
        spline=make_smoothing_spline(unique,mean_y,w=summed,lam=args.spline_lambda)
        spec=dict(kind='smoothing_spline',degree=int(spline.k),lam=args.spline_lambda,knots=spline.t.tolist(),coefficients=spline.c.tolist())
    model=FunctionApproximation(spec)
    definition=dict(schema='distance_function_v1',model_name='function_approximation',input_size=1,output_size=1,
        window_size=1,comparison_required_context=WINDOW_SIZE,distance_column=args.distance_column,scaler=scaler,function=spec,
        max_frame_gap_s=args.max_gap,recording_settings=fit[0]['settings'],fitted_trials=[g['name'] for g in fit],
        selection='Single fixed function fitted on Train only. No Test used for parameter selection.')
    (output/'models/function_approximation.json').write_text(json.dumps(definition,indent=2),encoding='utf-8')
    log(f'Fixed function: {args.function_kind}; no candidate selection.')
    return model,scaler,[],dict(model='function_approximation',completed_epochs=0,training_rule='Single fit; fixed function parameters')


class DistanceLSTM(nn.Module):
    """One distance feature per step; hidden/cell state reset for every window.

    Accept the same [batch, window] storage as MLP, then reshape to
    [batch, window, 1]. No hidden state crosses windows, trials or batches.
    """
    def __init__(self, hidden=HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(.10), nn.Linear(hidden, 1))

    def forward(self, x):
        sequence, _ = self.lstm(x.unsqueeze(-1))
        return self.head(sequence[:, -1, :]).squeeze(-1)


class ContinuousDistanceLSTM(nn.Module):
    """Causal many-to-many LSTM. One state per trial, never shared across trials."""
    def __init__(self, hidden=HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(.10), nn.Linear(hidden, 1))

    def forward(self, x, state=None):
        sequence, state = self.lstm(x, state)
        return self.head(sequence).squeeze(-1), state


def continuous_segments(distance, target, timestamp, frame, valid, max_gap):
    """Break only on missing input, bad timing or missing source frames."""
    usable = np.isfinite(distance) & np.isfinite(timestamp) & np.isfinite(frame) & valid
    segments, current = [], []
    for i in range(len(distance)):
        if not usable[i]:
            if current:segments.append(np.asarray(current));current=[]
            continue
        if current:
            j=current[-1]
            if frame[i]-frame[j]!=1 or not 0<timestamp[i]-timestamp[j]<=max_gap:
                segments.append(np.asarray(current));current=[]
        current.append(i)
    if current:segments.append(np.asarray(current))
    return [dict(distance=distance[idx].astype(np.float32), y=target[idx].astype(np.float32),
                 frame=frame[idx].astype(int), time=timestamp[idx]-timestamp[0]) for idx in segments]


def continuous_predictions(model, trial, scaler, device):
    model.eval()
    result=[]
    with torch.inference_mode():
        for seq in trial['sequences']:
            x=torch.tensor((seq['distance']-scaler['x_min'])/scaler['x_scale'],device=device).reshape(1,-1,1)
            pred,_=model(x)  # No state supplied: zero state at this trial/segment start.
            pred=pred[0].cpu().numpy()*scaler['y_scale']+scaler['y_mean']
            result.extend(dict(frame=int(f),time=float(t),actual=float(y),predicted=float(p))
                          for f,t,y,p in zip(seq['frame'],seq['time'],seq['y'],pred))
    return result


def sequence_epoch(model, loader, optimizer, loss_name, device, y_scale):
    from torch.nn.utils.rnn import pad_sequence
    model.train()
    total_loss=total_mae=count=0
    for batch in loader:
        xs,ys,weights=[],[],[]
        for sequences,scaler in batch:
            n=sum(int(np.isfinite(seq['y']).sum()) for seq in sequences)
            for seq in sequences:
                mask=np.isfinite(seq['y'])
                xs.append(torch.tensor((seq['distance']-scaler['x_min'])/scaler['x_scale']).reshape(-1,1))
                ys.append(torch.tensor(np.where(mask,(seq['y']-scaler['y_mean'])/scaler['y_scale'],0)))
                weights.append(torch.tensor(mask.astype(np.float32)/n/len(batch)))
        x=pad_sequence(xs,batch_first=True).to(device)
        y=pad_sequence(ys,batch_first=True).to(device)
        weight=pad_sequence(weights,batch_first=True).to(device)
        optimizer.zero_grad(set_to_none=True)
        pred,_=model(x)
        error=pred-y
        loss=((error.abs() if loss_name=='mae' else error.square())*weight).sum()
        loss.backward();optimizer.step()
        total_loss+=float(loss.detach())*len(batch)
        total_mae+=float((error.detach().abs()*weight).sum())*y_scale*len(batch)
        count+=len(batch)
    return total_loss/count,total_mae/count


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


def load_trials(source, max_gap, distance_column=DISTANCE_COLUMN):
    trials = []
    for folder in grasp_folders(source):
        with (folder / 'camera.csv').open(encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        if not rows or distance_column not in rows[0]:
            raise ValueError(f'{folder.name}: missing {distance_column}; no automatic distance fallback.')
        d, y, t, frame = [np.array([numeric(r, key) for r in rows]) for key in
                           (distance_column, 'finger_width', 't_camera', 'frame')]
        valid = np.array([r.get('valid') == '1' for r in rows])
        sequences=continuous_segments(d,y,t,frame,valid,max_gap)
        windows, indices = make_windows(d, y, t, frame, valid, max_gap)
        if not len(indices):
            raise ValueError(f'{folder.name}: no valid continuous {WINDOW_SIZE}-frame window.')
        meta = json.loads((folder / 'segment.json').read_text(encoding='utf-8-sig'))
        settings = json.loads((folder / 'settings.json').read_text(encoding='utf-8-sig'))
        phase = np.where(frame[indices] < meta['onset'], 'rest',
                         np.where(frame[indices] < meta['stable'], 'movement', 'hold'))
        trials.append(dict(name=folder.name, windows=windows, y=y[indices].astype(np.float32),
                           frame=frame[indices].astype(int), time=t[indices] - t[0],
                           window_start_frame=frame[indices - WINDOW_SIZE + 1].astype(int),
                           window_duration=t[indices] - t[indices - WINDOW_SIZE + 1],
                           phase=phase, raw_rows=len(rows), settings=settings,sequences=sequences))
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


def fit_sequence_scaler(trials):
    sequences=[s for g in trials for s in g['sequences']]
    x=np.concatenate([s['distance'] for s in sequences])
    y=np.concatenate([s['y'][np.isfinite(s['y'])] for s in sequences])
    scale=float(np.max(x)-np.min(x))
    if scale<=0:raise ValueError('Training distances have zero range.')
    return dict(x_min=float(x.min()),x_scale=scale,y_mean=float(y.mean()),y_scale=max(float(y.std()),1e-6))


def inputs_for(trial, inputs, scaler):
    x = trial['windows'] if inputs == WINDOW_SIZE else trial['windows'][:, -1:]
    return ((x - scaler['x_min']) / scaler['x_scale']).astype(np.float32)


def loader_for(trials, inputs, scaler, args):
    if inputs == 0:
        generator=torch.Generator().manual_seed(args.seed)
        return DataLoader([(g['sequences'],scaler) for g in trials],batch_size=args.batch_size,
                          shuffle=True,generator=generator,collate_fn=lambda batch:batch)
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
    if isinstance(model,FunctionApproximation):
        x=(trial['windows'][:,-1]-scaler['x_min'])/scaler['x_scale']
        return model.predict(x)*scaler['y_scale']+scaler['y_mean']
    if isinstance(model,ContinuousDistanceLSTM):
        values={r['frame']:r['predicted'] for r in continuous_predictions(model,trial,scaler,device)}
        return np.array([values[int(f)] for f in trial['frame']])
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


def test_mae(model, trials, inputs, scaler, device):
    return float(np.mean([np.mean(abs(predict_trial(model, g, inputs, scaler, device) - g['y']))
                          for g in trials]))


def epoch_step(model, loader, optimizer, loss_name, device, y_scale):
    if isinstance(model,ContinuousDistanceLSTM):
        return sequence_epoch(model,loader,optimizer,loss_name,device,y_scale)
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
    if name == LSTM_MODEL:
        return dict(schema='distance_lstm_continuous_v1',model_name=name,architecture='LSTM',
            state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
            input_size=1,feature_size=1,output_size=1,hidden_size=args.hidden,lstm_layers=1,dropout=.1,
            window_size=1,sequence_length='Full trial',comparison_required_context=WINDOW_SIZE,
            state_policy='Retain state from trial start to current frame; reset at new trial or invalid/gapped input',
            input_order='Chronological distance sequence from trial start; no future or aperture input',
            target='Current-frame fingertip aperture at every valid labelled time step',
            parameter_count=sum(p.numel() for p in model.parameters()),
            scaler=scaler,loss=args.loss,epochs=epoch,seed=args.seed,
            distance_column=args.distance_column,max_frame_gap_s=args.max_gap,
            recording_settings=trials[0]['settings'],fitted_trials=[g['name'] for g in trials])
    if name == WINDOW_LSTM_MODEL:
        saved = dict(schema='distance_lstm_v1', model_name=name, architecture='LSTM',
            state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
            input_size=1, feature_size=1, output_size=1, hidden_size=args.hidden, lstm_layers=1, dropout=.1,
            window_size=inputs, sequence_length=inputs, comparison_required_context=WINDOW_SIZE,
            state_policy='Zero hidden/cell state independently for every sliding window; never carry state',
            input_order='Oldest to current distance, one scalar per recurrent step',
            target='Current-frame fingertip aperture; no aperture input',
            parameter_count=sum(p.numel() for p in model.parameters()), scaler=scaler, loss=args.loss,
            epochs=epoch, seed=args.seed, distance_column=args.distance_column, max_frame_gap_s=args.max_gap,
            recording_settings=trials[0]['settings'], fitted_trials=[g['name'] for g in trials])
        return saved
    return dict(schema='distance_mlp_v1', model_name=name,
                architecture='MLP', feature_size=1, sequence_length=1,
                lstm_layers=0, state_policy='Stateless',
                output_size=1, parameter_count=sum(p.numel() for p in model.parameters()),
                state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                input_size=inputs, hidden_size=args.hidden, dropout=.10,
                window_size=inputs, comparison_required_context=WINDOW_SIZE,
                input_order='oldest to newest; current frame included',
                target='current-frame fingertip aperture; no historical aperture inputs',
                scaler=scaler, loss=args.loss, epochs=epoch, seed=args.seed,
                distance_column=args.distance_column,
                max_frame_gap_s=args.max_gap, recording_settings=trials[0]['settings'],
                fitted_trials=[g['name'] for g in trials])


def train_one(name, inputs, fit_trials, test_trials, args, output, device, log):
    # Test values never enter fitting, early stopping, checkpoint selection or normalization.
    if name=='function_approximation':return fit_fixed_function(fit_trials,args,output,log)
    seed_all(args.seed)
    scaler=fit_sequence_scaler(fit_trials) if name==LSTM_MODEL else fit_scaler(fit_trials)
    model=(ContinuousDistanceLSTM(args.hidden) if name==LSTM_MODEL else DistanceLSTM(args.hidden) if name==WINDOW_LSTM_MODEL else DistanceMLP(inputs,args.hidden)).to(device)
    loader=loader_for(fit_trials,inputs,scaler,args)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    history=[]
    for epoch in range(1,args.epochs+1):
        loss,mae=epoch_step(model,loader,optimizer,args.loss,device,scaler['y_scale'])
        history.append(dict(model=name,stage='fixed_epochs',epoch=epoch,train_loss=loss,train_mae_px=mae))
        if epoch==1 or epoch%args.log_every==0 or epoch==args.epochs:
            log(f'[{name}] epoch {epoch}/{args.epochs} | Train MAE={mae:.3f}px')
    model.eval()
    saved=checkpoint(model,name,inputs,scaler,args,args.epochs,fit_trials)
    saved['training_rule']='Fixed epoch count; save last epoch only; no early stopping or Test-based selection'
    torch.save(saved,output/'models'/f'{name}.pth')
    log(f'[{name}] Saved LAST epoch {args.epochs}; Test not evaluated during training.')
    return model,scaler,history,dict(model=name,completed_epochs=args.epochs,training_rule='Fixed epochs; last checkpoint')


def load_folder_partitions(source, max_gap, distance_column=DISTANCE_COLUMN):
    import hashlib
    train_dir, test_dir = source / TRAIN_FOLDER, source / TEST_FOLDER
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise ValueError(f'Required folders: {train_dir} and {test_dir}')
    train, test = load_trials(train_dir, max_gap, distance_column), load_trials(test_dir, max_gap, distance_column)
    if not train or not test:
        raise ValueError('Train and Test must both contain grasp trial folders.')
    if {g['name'] for g in train} & {g['name'] for g in test}:
        raise ValueError('Duplicate trial names across Train and Test.')
    hashes = [{hashlib.sha256((folder / g['name'] / 'camera.csv').read_bytes()).hexdigest()
               for g in group} for folder,group in [(train_dir,train),(test_dir,test)]]
    if hashes[0] & hashes[1]:
        raise ValueError('Identical camera.csv data exists in both Train and Test.')
    signatures = {(g['settings'].get('object_color','black'),g['settings']['size_cm'],
                   g['settings']['width'],g['settings']['height']) for g in train+test}
    if len(signatures) != 1:
        raise ValueError('Train and Test object classes/resolutions differ.')
    manifest=source/'random_split_manifest.csv'
    if manifest.is_file():
        with manifest.open(encoding='utf-8-sig',newline='') as f:
            conditions={r['trial']:r['condition'] for r in csv.DictReader(f)}
        for g in train+test:
            if g['name'] in conditions:g['condition_label']=conditions[g['name']]
    return train,test


def random_trial_partitions(source, max_gap, distance_column, seed, test_fraction):
    import hashlib
    folders=sorted(grasp_folders(source),key=lambda p:p.name)
    trials=load_trials(source,max_gap,distance_column)
    if len(trials)<2:raise ValueError('Random split requires at least two complete trials.')
    hashes=[hashlib.sha256((folder/'camera.csv').read_bytes()).hexdigest() for folder in folders]
    if len(set(hashes)) != len(hashes):
        raise ValueError('Duplicate camera.csv files found; remove duplicate trials before splitting.')
    ordered=sorted(trials,key=lambda g:g['name'])
    random.Random(seed).shuffle(ordered)
    n_test=max(1,min(len(ordered)-1,int(len(ordered)*test_fraction+.5)))
    test=ordered[:n_test];train=ordered[n_test:]
    assert not ({g['name'] for g in train}&{g['name'] for g in test})
    return train,test


def configure_window(size):
    global WINDOW_SIZE, WINDOW_MODEL, PREDICTION_KEY, MODEL_SPECS, LABELS, LSTM_MODEL, LSTM_PREDICTION_KEY, WINDOW_LSTM_MODEL, WINDOW_LSTM_KEY
    WINDOW_SIZE = size
    WINDOW_MODEL = f'distance_window{size}'
    PREDICTION_KEY = f'predicted_window{size}_px'
    LSTM_MODEL = 'distance_lstm_continuous'
    LSTM_PREDICTION_KEY = 'predicted_lstm_continuous_px'
    WINDOW_LSTM_MODEL = f'distance_lstm_window{size}'
    WINDOW_LSTM_KEY = f'predicted_lstm_window{size}_px'
    MODEL_SPECS = {'function_approximation': 1, WINDOW_MODEL: size, WINDOW_LSTM_MODEL: size, LSTM_MODEL: 0}
    LABELS = {'function_approximation': 'Function approximation', WINDOW_MODEL: f'MLP ({size} frames)', WINDOW_LSTM_MODEL: f'LSTM ({size} frames)', LSTM_MODEL: 'LSTM (full history)'}


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=SOURCE)
    parser.add_argument('--split-mode', choices=['auto','random','folders'], default='auto', help='Auto uses Train/Test when both exist; otherwise randomly splits whole trials.')
    parser.add_argument('--test-fraction', type=float, default=TEST_FRACTION)
    parser.add_argument('--check-data', action='store_true', help='Validate splits and forward shapes only; never train or save results')
    parser.add_argument('--window-size', type=int, default=WINDOW_SIZE, help='History length including current frame')
    parser.add_argument('--distance-column', choices=['hand_object_signed_distance', 'hand_object_distance'], default=DISTANCE_COLUMN)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--function-kind', choices=['smoothing_spline','polynomial'], default=FUNCTION_KIND)
    parser.add_argument('--spline-lambda', type=float, default=SPLINE_LAMBDA)
    parser.add_argument('--poly-degree', type=int, choices=range(3,10), default=POLY_DEGREE)
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
    if not 0 < args.test_fraction < 1:
        parser.error('--test-fraction must be between 0 and 1')
    if args.window_size < 2:
        parser.error('--window-size must be at least 2')
    configure_window(args.window_size)
    text = str(args.source).strip() or input('请输入数据文件夹绝对路径：\n> ').strip()
    source = Path(text.strip('\"\''))
    if not source.is_absolute() or not source.is_dir():
        parser.error('请输入存在的数据文件夹绝对路径。')
    if (args.epochs < 1 or not np.isfinite(args.spline_lambda) or args.spline_lambda < 0
            or args.batch_size < 1 or args.hidden < 1 or args.lr <= 0 or args.max_gap <= 0
            or args.log_every < 1 or args.weight_decay < 0):
        parser.error('Invalid counts or training parameters.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --device cpu.')
    device = torch.device('cuda' if args.device == 'cuda' or
                          (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    split_mode=args.split_mode
    if split_mode=='auto':
        split_mode='folders' if (source/TRAIN_FOLDER).is_dir() and (source/TEST_FOLDER).is_dir() else 'random'
    try:
        if split_mode=='random':
            fit_trials,val_trials=random_trial_partitions(source,args.max_gap,args.distance_column,args.seed,args.test_fraction)
        else:
            fit_trials,val_trials=load_folder_partitions(source,args.max_gap,args.distance_column)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    if args.check_data:
        check_models_and_data(fit_trials, val_trials, args)
        return
    trials = fit_trials + val_trials
    test = val_trials
    output = args.output or source / 'Result' / f'fixed_four_models_window{WINDOW_SIZE}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    if output.exists():
        parser.error(f'Output already exists: {output}')
    (output / 'models').mkdir(parents=True)
    (output / 'test_predictions').mkdir()
    def log(message=''):
        print(message, flush=True)
        with (output / 'training_log.txt').open('a', encoding='utf-8') as f:
            f.write(message + '\n')
    started = time.perf_counter()
    log(f'=== Single-point function approximation vs {WINDOW_SIZE}-frame MLP vs {WINDOW_SIZE}-frame LSTM vs continuous-history LSTM ===')
    log(f'Data: {source}\nDevice: {device} | Loss: {args.loss} | Seed: {args.seed}')
    log(f'Input distance column: {args.distance_column}')
    log(f'Train: {len(fit_trials)} trials | Split: {split_mode} | Source: {source}')
    log(f'Test: {len(val_trials)} trials | Seed: {args.seed}')
    log('Fixed epochs and fixed function parameters. Test evaluated only after all fitting is complete.')
    log(f'All four models predict the SAME target frames; first {WINDOW_SIZE-1} frames and discontinuous windows are omitted.')
    log(f'Fixed training epochs: {args.epochs}; no early stopping; last epoch saved.')
    log(f'Results: {output.resolve()}')
    split = dict(seed=args.seed, strategy='Random complete-trial split' if split_mode=='random' else 'Fixed Train/Test folders',
                 split_mode=split_mode, requested_test_fraction=args.test_fraction if split_mode=='random' else None,
                 source_directory=str(source),
                 train_directory=str(source if split_mode=='random' else source/TRAIN_FOLDER),
                 test_directory=str(source if split_mode=='random' else source/TEST_FOLDER),
                 train_trials=[g['name'] for g in fit_trials],
                 test_trials=[g['name'] for g in val_trials],
                 selection='Fixed epochs, last checkpoint; fixed function; Train-only normalization; Test never selects models.')
    (output / 'split.json').write_text(json.dumps(split, indent=2), encoding='utf-8')
    split_rows = []
    val_names = {g['name'] for g in val_trials}
    for g in trials:
        split_rows.append(dict(trial=g['name'], split='test' if g['name'] in val_names else 'train',
            internal_role='test' if g['name'] in val_names else 'fit',
            raw_rows=g['raw_rows'], eligible_target_frames=len(g['y']),
            omitted_rows=g['raw_rows'] - len(g['y'])))
    write_csv(output / 'data_split.csv', split_rows)
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings.update(source=str(source), output=str(output), window_size=WINDOW_SIZE,
                    normalization='Input MinMax and output standardization, fitted only on the fitting partition at each stage.',
                    window_rule=f'{WINDOW_SIZE} consecutive valid source frames, increasing actual timestamps, no gap over max_gap.',
                    models=MODEL_SPECS,
                    model_information=model_descriptions(args.hidden),
                    window_lstm_state='Zero state independently for each window; one distance feature per step.',
                    lstm_state='Full trial from first valid frame to current frame; reset at trial boundary or invalid/gapped input.',
                    lstm_training='Full-sequence backpropagation; batch_size counts trials; padded target positions masked; each trial has equal total loss weight.',
                    lstm_comparison='Main comparison uses the same eligible MLP target frames; full LSTM trajectories and metrics also exported separately.')
    (output / 'model_information.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    trained, histories, selections = {}, [], []
    for name, inputs in MODEL_SPECS.items():
        model, scaler, history, selection = train_one(name, inputs, fit_trials, val_trials,
                                                      args, output, device, log)
        trained[name] = (model, scaler)
        histories.extend(history)
        selections.append(selection)
    write_csv(output / 'training_history.csv', histories)
    write_csv(output / 'training_completion.csv', selections)
    log('\n=== Test evaluation: identical target frames for all four models ===')
    metrics_rows, all_predictions = [], []
    pooled = {name: [] for name in MODEL_SPECS}
    truth = []
    for g in test:
        predictions = {name: predict_trial(model, g, MODEL_SPECS[name], scaler, device)
                       for name, (model, scaler) in trained.items()}
        truth.append(g['y'])
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(g['time'], g['y'], color='#222222', lw=2, label='Measured')
        for name, color in zip(MODEL_SPECS, ['#3485b8', '#e59a32', '#9565b0', '#519e58']):
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
                predicted_function_px=float(predictions['function_approximation'][i]),
                **{PREDICTION_KEY: float(predictions[WINDOW_MODEL][i]),
                   WINDOW_LSTM_KEY: float(predictions[WINDOW_LSTM_MODEL][i]),
                   LSTM_PREDICTION_KEY: float(predictions[LSTM_MODEL][i])})
            trial_rows.append(row)
        all_predictions.extend(trial_rows)
        write_csv(output / 'test_predictions' / f'{g["name"]}.csv', trial_rows)
        ax.set(title=g['name'], xlabel='Time from segment start (s)', ylabel='Aperture (px)')
        ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
        fig.savefig(output / 'test_predictions' / f'{g["name"]}.png', dpi=160); plt.close(fig)
    write_csv(output / 'test_trial_metrics.csv', metrics_rows)
    write_csv(output / 'test_predictions.csv', all_predictions)
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
        for name, key in prediction_keys().items():
            phase_rows.append(dict(model=name, phase=phase, frames=len(subset),
                **scores(actual, np.array([r[key] for r in subset]))))
    write_csv(output / 'test_phase_metrics.csv', phase_rows)
    export_phase_errors(output, all_predictions, prefix='test')
    export_full_lstm_results(output, test, trained[LSTM_MODEL], device)
    plot_results(output, summary, metrics_rows, histories, test)
    export_condition_results(output, all_predictions, test)
    log('\n=== Overall test metrics (all eligible test frames pooled) ===')
    for row in summary:
        r2 = f'{row["r2"]:.3f}' if row['r2'] is not None else 'N/A'
        log(f'{LABELS[row["model"]]}: MAE={row["mae"]:.2f}px | RMSE={row["rmse"]:.2f}px | R2={r2}')
    log(f'Total time: {time.perf_counter() - started:.1f}s')
    log(f'\nSaved results to: {output.resolve()}')
    for name in ['test_model_comparison.png', 'test_metrics_chart.png', 'test_predictions/*.png',
                 'training_curves.png', 'overall_model_summary.csv', 'test_phase_metrics.csv',
                 'test_phase_errors.png', 'test_trial_phase_metrics.csv', 'test_phase_summary.csv',
                 'data_split.csv', 'test_condition_metrics.csv', 'test_metrics_by_condition.png',
                 'models/function_approximation.json', f'models/{WINDOW_MODEL}.pth', f'models/{WINDOW_LSTM_MODEL}.pth', f'models/{LSTM_MODEL}.pth']:
        log('  ' + name)
    (output / 'RESULT.txt').write_text((output / 'training_log.txt').read_text(encoding='utf-8'), encoding='utf-8')


def prediction_keys():
    return {'function_approximation': 'predicted_function_px', WINDOW_MODEL: PREDICTION_KEY,
            WINDOW_LSTM_MODEL: WINDOW_LSTM_KEY, LSTM_MODEL: LSTM_PREDICTION_KEY}


def export_full_lstm_results(output,trials,trained,device):
    model,scaler=trained
    dest=output/'test_lstm_full_predictions';dest.mkdir()
    metrics=[]
    for g in trials:
        rows=continuous_predictions(model,g,scaler,device)
        write_csv(dest/f'{g["name"]}.csv',rows)
        y=np.array([r['actual'] for r in rows]);p=np.array([r['predicted'] for r in rows])
        valid=np.isfinite(y)&np.isfinite(p)
        if valid.any():metrics.append(dict(trial=g['name'],condition=condition_label(g),frames=int(valid.sum()),**scores(y[valid],p[valid])))
        fig,ax=plt.subplots(figsize=(11,4.5))
        # Separate plotted segments prevent a line bridging a detection gap.
        by_frame={r['frame']:r for r in rows}
        for j,seq in enumerate(g['sequences']):
            subset=[by_frame[int(f)] for f in seq['frame']]
            ax.plot([r['time'] for r in subset],[r['actual'] for r in subset],color='#222222',label='Measured' if j==0 else None)
            ax.plot([r['time'] for r in subset],[r['predicted'] for r in subset],color='#519e58',label='LSTM (full history)' if j==0 else None)
        ax.set(title=g['name'],xlabel='Time from segment start (s)',ylabel='Aperture (px)');ax.legend();ax.grid(alpha=.2);fig.tight_layout()
        fig.savefig(dest/f'{g["name"]}.png',dpi=160);plt.close(fig)
    write_csv(output/'test_lstm_full_trial_metrics.csv',metrics)


def export_phase_errors(output, predictions, prefix='test'):
    phases = [('rest', 'Initial'), ('movement', 'Grasping'), ('hold', 'Hold')]
    per_trial, summary = [], []
    for name, key in prediction_keys().items():
        for phase, _ in phases:
            for trial in sorted({r['trial'] for r in predictions}):
                rows = [r for r in predictions if r['trial']==trial and r['phase']==phase]
                if rows:
                    per_trial.append(dict(model=name,trial=trial,phase=phase,frames=len(rows),
                        mae_px=float(np.mean([abs(r[key]-r['actual_aperture_px']) for r in rows]))))
            part = [r for r in per_trial if r['model']==name and r['phase']==phase]
            values = [r['mae_px'] for r in part]
            summary.append(dict(model=name,phase=phase,grasps=len(values),frames=sum(r['frames'] for r in part),
                mean_trial_mae_px=float(np.mean(values)) if values else None,
                sd_trial_mae_px=float(np.std(values,ddof=1)) if len(values)>1 else 0. if values else None))
    write_csv(output/f'{prefix}_trial_phase_metrics.csv',per_trial)
    write_csv(output/f'{prefix}_phase_summary.csv',summary)
    fig,ax=plt.subplots(figsize=(9,5.5))
    for j,(name,color) in enumerate(zip(MODEL_SPECS,['#3485b8','#e59a32','#9565b0','#519e58'])):
        rows=[r for r in summary if r['model']==name]
        ax.bar(np.arange(3)+(j-1.5)*.20,[r['mean_trial_mae_px'] if r['mean_trial_mae_px'] is not None else np.nan for r in rows],width=.18,
               yerr=[r['sd_trial_mae_px'] or 0 for r in rows],capsize=4,color=color,label=LABELS[name])
    ax.set_xticks(range(3),[label for _,label in phases]);ax.set_ylabel('Mean per-grasp MAE (px)')
    ax.legend();ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(output/f'{prefix}_phase_errors.{ext}',dpi=200)
    plt.close(fig)
    return summary



def plot_results(output, summary, per_trial, histories, test):
    colors = ['#3485b8', '#e59a32', '#9565b0', '#519e58']
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, key, title in zip(axs, ['mae', 'rmse', 'r2'], ['MAE (px)', 'RMSE (px)', 'R\u00b2']):
        values = [r[key] if r[key] is not None else np.nan for r in summary]
        bars = ax.bar(list(LABELS.values()), values, color=colors)
        ax.bar_label(bars, fmt='%.3f' if key == 'r2' else '%.2f', padding=3)
        ax.tick_params(axis='x',labelsize=8); ax.set_title(title); ax.margins(y=.2); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('Same test trials and target frames')
    fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(output / 'test_model_comparison.png', dpi=180)
    fig.savefig(output / 'test_model_comparison.pdf'); plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(14, max(4.5, .45 * (len(test) + 1) + 2)))
    pos = np.arange(len(test) + 1)
    for ax, key, title in zip(axs, ['mae', 'rmse', 'r2'], ['MAE (px)', 'RMSE (px)', 'R\u00b2']):
        for j, (name, color) in enumerate(zip(MODEL_SPECS, colors)):
            by_trial = {r['trial']: r for r in per_trial if r['model'] == name}
            overall = next(r for r in summary if r['model'] == name)
            values = [by_trial[g['name']][key] for g in test] + [overall[key]]
            values = [v if v is not None else np.nan for v in values]
            ax.barh(pos + (j - 1.5) * .20, values, height=.18, color=color, label=LABELS[name])
        ax.set_yticks(pos, [f'Test {i:02d}' for i in range(1, len(test) + 1)] + ['Overall'])
        ax.invert_yaxis(); ax.set_title(title); ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
    axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output / 'test_metrics_chart.png', dpi=170)
    fig.savefig(output / 'test_metrics_chart.pdf'); plt.close(fig)
    write_csv(output / 'test_label_mapping.csv', [dict(label=f'Test {i:02d}', trial=g['name']) for i, g in enumerate(test, 1)])
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax,name in zip(axs,[WINDOW_MODEL,WINDOW_LSTM_MODEL,LSTM_MODEL]):
        rows=[r for r in histories if r['model']==name]
        ax.plot([r['epoch'] for r in rows],[r['train_mae_px'] for r in rows],label='Train (dropout active)')
        ax.set(title=LABELS[name],xlabel='Epoch',ylabel='MAE (px)');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/'training_curves.png',dpi=160);plt.close(fig)



def model_descriptions(hidden):
    result = {}
    for name, inputs in MODEL_SPECS.items():
        if name == 'function_approximation':
            result[name]=dict(architecture='Single-point function approximation',input_units=1,output_units=1,
                input_correspondence='Current signed distance only',output_correspondence='Current-frame finger_width (px)',
                selection='Single function with predeclared type/parameters; Train only',
                objective='Squared-error fitting plus spline roughness penalty; neural models retain MAE loss')
            continue
        lstm = name in (WINDOW_LSTM_MODEL, LSTM_MODEL)
        model = ContinuousDistanceLSTM(hidden) if name == LSTM_MODEL else DistanceLSTM(hidden) if lstm else DistanceMLP(inputs, hidden)
        result[name] = dict(architecture='LSTM' if lstm else 'MLP',
            input_units=1 if lstm else inputs, sequence_steps='From trial start to current frame' if name==LSTM_MODEL else WINDOW_SIZE if lstm else 1,
            input_correspondence='One distance feature at each step, oldest to current' if lstm else
                                 'Current-frame distance only' if inputs == 1 else 'One distance per unit, oldest to current',
            output_units=1, output_correspondence='Current-frame finger_width (px)',
            hidden_layers=[hidden] if lstm else [hidden, hidden],
            parameter_count=sum(p.numel() for p in model.parameters()),
            state_reset='Trial start or missing/gapped input; never reset at normal sliding-window boundaries' if name==LSTM_MODEL else 'Zero state for every window' if lstm else 'Stateless')
    return result


def condition_label(trial):
    if 'condition_label' in trial:return trial['condition_label']
    settings = trial['settings']
    guide = settings.get('approach_guide') or settings.get('metronome_at_recording_start') or {}
    target = guide.get('target_movement_s')
    return f'{float(target):g} s target' if target is not None else 'Unknown condition'


def check_models_and_data(fit, test, args):
    print(f'CHECK ONLY: {len(fit)} Train, {len(test)} Test; {WINDOW_SIZE}-frame common context.')
    scaler = fit_scaler(fit)
    with torch.inference_mode():
        for name, inputs in MODEL_SPECS.items():
            if name == 'function_approximation':continue
            model = ContinuousDistanceLSTM(args.hidden) if name == LSTM_MODEL else DistanceLSTM(args.hidden) if name==WINDOW_LSTM_MODEL else DistanceMLP(inputs,args.hidden)
            if name == LSTM_MODEL:
                values = predict_trial(model,test[0],inputs,scaler,torch.device('cpu'))
                assert values.shape == test[0]['y'].shape
            else:
                x = torch.from_numpy(inputs_for(test[0],inputs,scaler)[:3])
                assert model(x).shape == (len(x),)
    for split, trials in [('Train',fit),('Test',test)]:
        counts = {}
        for g in trials:
            label=condition_label(g);counts[label]=counts.get(label,0)+1
        print(split, counts)
    print(json.dumps(model_descriptions(args.hidden),indent=2))
    print('Forward shapes verified. No training, no optimizer steps, no results written.')


def export_condition_results(output, predictions, trials):
    labels = {g['name']:condition_label(g) for g in trials}
    groups = sorted(set(labels.values()))
    report = []
    for group in groups+['Overall']:
        rows = predictions if group == 'Overall' else [r for r in predictions if labels[r['trial']]==group]
        y = np.array([r['actual_aperture_px'] for r in rows])
        for name,key in prediction_keys().items():
            report.append(dict(condition=group,model=name,trials=len({r['trial'] for r in rows}),frames=len(rows),
                               **scores(y,np.array([r[key] for r in rows]))))
    write_csv(output/'test_condition_metrics.csv',report)
    write_csv(output/'test_condition_mapping.csv',[dict(trial=name,condition=label) for name,label in labels.items()])
    fig,axes=plt.subplots(len(groups)+1,3,figsize=(13,3.4*(len(groups)+1)),squeeze=False)
    for row_axes,group in zip(axes,groups+['Overall']):
        rows=[r for r in report if r['condition']==group]
        for ax,key,title in zip(row_axes,['mae','rmse','r2'],['MAE (px)','RMSE (px)','R²']):
            bars=ax.bar(list(LABELS.values()),[r[key] if r[key] is not None else np.nan for r in rows],color=['#3485b8','#e59a32','#9565b0','#519e58'])
            ax.bar_label(bars,fmt='%.3f' if key=='r2' else '%.2f',padding=3)
            ax.set_title(f'{group}: {title}');ax.tick_params(axis='x',labelsize=7);ax.margins(y=.25);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(output/f'test_metrics_by_condition.{ext}',dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    main()
