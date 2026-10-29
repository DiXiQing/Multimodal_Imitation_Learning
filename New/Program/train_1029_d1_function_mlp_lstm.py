"""Blue 1029 D1: current-distance Function, MLP 5f, causal LSTM Full.

Run --check-data --smoke-export for validation WITHOUT fitting or training.
Run without these flags to copy a whole-trial split and train 400 fixed epochs.
Reference: train_fixed_function_mlp_lstm.py (left unchanged).
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
from datetime import datetime
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from scipy.interpolate import BSpline, make_smoothing_spline

SOURCE = Path(__file__).resolve().parents[2] / 'Data/1029_D1/20mm_red'
DISTANCE = 'index_mcp_object_distance_mm'
TARGET = 'finger_width_mm'
EPOCHS, WINDOW, SEED, MAX_GAP = 400, 5, 20261029, .10
LABELS = ('Function', 'MLP 5f', 'LSTM Full')
COLORS = ('#3485b8', '#e59a32', '#519e58')
SPLINE_LAMBDA = 1e-6  # Predeclared; never chosen using Test.


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


def folders(parent):
    # Deliberately non-recursive. backup, Result and other derived folders never enter.
    return sorted(p for p in parent.iterdir() if p.is_dir() and re.fullmatch(r'grasp_\d+', p.name))


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
    for mm, px in [(DISTANCE, 'index_mcp_object_distance'), (TARGET, 'finger_width')]:
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
    seqs = sequence_indices(d, y, t, frame, valid)
    windows, indices = [], []
    for idx in seqs:
        for k in range(WINDOW - 1, len(idx)):
            windows.append(d[idx[k-WINDOW+1:k+1]]); indices.append(idx[k])
    if not indices: raise ValueError(f'{folder}: no valid five-frame common targets')
    return dict(name=folder.name, folder=folder, rows=rows, fields=fields, settings=settings,
                d=d, y=y, t=t-t[0], unix=t, frame=frame.astype(np.int64), seqs=seqs,
                windows=np.array(windows), common=np.array(indices), csv_sha256=file_hash(required[0]))


def validate_partitions(train, test):
    if not train or not test: raise ValueError('Train and Test must each contain experiments')
    if {g['name'] for g in train} & {g['name'] for g in test}:
        raise ValueError('Same experiment name occurs in both Train and Test')
    if {g['csv_sha256'] for g in train} & {g['csv_sha256'] for g in test}:
        raise ValueError('Identical CSV occurs in Train and Test')
    # Absolute timestamps + source frame identify recorded samples even after a folder rename.
    keys = lambda trials: {(int(f), float(t)) for g in trials for f, t in zip(g['frame'], g['unix']) if np.isfinite(t)}
    if keys(train) & keys(test): raise ValueError('Overlapping recorded frames across Train/Test')
    signatures = {(g['settings'].get('object_color'), g['settings'].get('size_cm'),
                   g['settings'].get('width'), g['settings'].get('height')) for g in train+test}
    if len(signatures) != 1: raise ValueError('Mixed object classes or image resolutions')


def get_partitions(source, persist=False):
    train_dir, test_dir = source/'Train', source/'Test'
    if train_dir.exists() != test_dir.exists():
        raise ValueError('Only one of Train/Test exists; refusing to invent or overwrite the split')
    existing = train_dir.exists()
    if existing:
        train = [load_trial(p) for p in folders(train_dir)]
        test = [load_trial(p) for p in folders(test_dir)]
    else:
        all_trials = [load_trial(p) for p in folders(source)]
        if len(all_trials) < 2: raise ValueError('Need at least two experiments')
        order = list(range(len(all_trials))); random.Random(SEED).shuffle(order)
        n = max(1, min(len(order)-1, int(len(order)*.75)))
        train = sorted([all_trials[i] for i in order[:n]], key=lambda g:g['name'])
        test = sorted([all_trials[i] for i in order[n:]], key=lambda g:g['name'])
    validate_partitions(train, test)
    manifest = dict(seed=SEED, strategy='existing Train/Test reused' if existing else 'whole-experiment random 75/25',
                    train=[g['name'] for g in train], test=[g['name'] for g in test],
                    records=[dict(split=s, trial=g['name'], source=str(g['folder']), csv_sha256=g['csv_sha256'])
                             for s, group in [('Train', train), ('Test', test)] for g in group])
    if persist and not existing:
        # Staging avoids leaving half-built Train/Test after a failed copy.
        stage = source/'.split_staging_1029'
        if stage.exists(): raise ValueError(f'Prior incomplete split exists: {stage}; inspect before rerun')
        stage.mkdir()
        for split, group in [('Train', train), ('Test', test)]:
            (stage/split).mkdir()
            for g in group:
                target = stage/split/g['name']; shutil.copytree(g['folder'], target)
                for f in g['folder'].rglob('*'):
                    if f.is_file() and file_hash(f) != file_hash(target/f.relative_to(g['folder'])):
                        raise ValueError(f'Copy verification failed: {f}')
        (stage/'Train').rename(train_dir); (stage/'Test').rename(test_dir); stage.rmdir()
    if persist: write_json(source/'split_manifest.json', manifest)
    return train, test, manifest


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5,32), nn.ReLU(), nn.Dropout(.1),
                                 nn.Linear(32,32), nn.ReLU(), nn.Linear(32,1))
    def forward(self, x): return self.net(x).squeeze(-1)


class FullLSTM(nn.Module):
    def __init__(self):
        super().__init__(); self.lstm = nn.LSTM(1,32,batch_first=True)
        self.head = nn.Sequential(nn.Dropout(.1),nn.Linear(32,1))
    def forward(self, x, state=None):
        hidden, state = self.lstm(x, state)
        return self.head(hidden).squeeze(-1), state


def fit_scaler(train):
    # Numerical conditioning only: one shared Train-only scaler, no per-trial normalization.
    x = np.concatenate([g['d'][idx] for g in train for idx in g['seqs']])
    y = np.concatenate([g['y'][g['common']] for g in train])
    return dict(x_mean=float(x.mean()), x_std=max(float(x.std()),1e-6),
                y_mean=float(y.mean()), y_std=max(float(y.std()),1e-6))


def tx(x, scale, device):
    return torch.tensor((x-scale['x_mean'])/scale['x_std'], dtype=torch.float32, device=device)


class FixedFunction:
    def __init__(self, definition): self.definition = definition
    def predict(self, x):
        s = self.definition
        if s['kind'] == 'mock_constant': return np.zeros_like(x)
        return BSpline(s['knots'],s['coefficients'],s['degree'],extrapolate=True)(x)


def fit_function(train, scale):
    x = np.concatenate([g['d'][g['common']] for g in train]); y = np.concatenate([g['y'][g['common']] for g in train])
    x = (x-scale['x_mean'])/scale['x_std']; y = (y-scale['y_mean'])/scale['y_std']
    weights = np.concatenate([np.full(len(g['common']),1/len(g['common'])) for g in train])
    unique, inv = np.unique(x,return_inverse=True)
    if len(unique)<5: raise ValueError('Function needs >=5 unique Train distances')
    w = np.bincount(inv,weights=weights); mean = np.bincount(inv,weights=weights*y)/w
    spline = make_smoothing_spline(unique,mean,w=w,lam=SPLINE_LAMBDA)
    return FixedFunction(dict(kind='smoothing_spline',degree=int(spline.k),knots=spline.t.tolist(),
                              coefficients=spline.c.tolist(),lambda_fixed=SPLINE_LAMBDA))


def train_networks(train, scale, device):
    # No Test argument exists: Test cannot affect fitting, curves, or epoch selection.
    models = {'MLP 5f':MLP().to(device), 'LSTM Full':FullLSTM().to(device)}
    history = []
    for label, model in models.items():
        optimizer = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
        rng = random.Random(SEED)
        for epoch in range(1,EPOCHS+1):
            model.train(); loss_sum = 0.
            order=list(range(len(train)));rng.shuffle(order)
            # Each optimizer step is one complete experiment; equal trial loss weights.
            for number in order:
                g=train[number];optimizer.zero_grad(set_to_none=True)
                if label == 'MLP 5f':
                    pred=model(tx(g['windows'],scale,device))
                    y=torch.tensor((g['y'][g['common']]-scale['y_mean'])/scale['y_std'],dtype=torch.float32,device=device)
                    loss=(pred-y).abs().mean()
                else:
                    loss=torch.zeros((),device=device)
                    # Fresh zero state per trial AND per invalid/gapped sequence. Entire
                    # continuous prefix participates in full BPTT, never detached/truncated.
                    for idx in g['seqs']:
                        pred,_=model(tx(g['d'][idx],scale,device).reshape(1,-1,1))
                        mask=np.isin(idx,g['common'])
                        if not mask.any(): continue
                        target=torch.tensor((g['y'][idx[mask]]-scale['y_mean'])/scale['y_std'],dtype=torch.float32,device=device)
                        loss=loss+(pred[0,torch.tensor(mask,device=device)]-target).abs().sum()/len(g['common'])
                loss.backward();optimizer.step();loss_sum+=float(loss.detach())*scale['y_std']
            history.append(dict(model=label,epoch=epoch,train_mae_mm=loss_sum/len(train)))
            if epoch==1 or epoch%25==0:print(f'{label}: epoch {epoch}/{EPOCHS}, Train MAE {history[-1]["train_mae_mm"]:.4f} mm',flush=True)
    return models, history


def predict(g, models, function, scale, device):
    out={label:np.full(len(g['d']),np.nan) for label in LABELS}
    idx=g['common']; x=(g['d'][idx]-scale['x_mean'])/scale['x_std']
    out['Function'][idx]=function.predict(x)*scale['y_std']+scale['y_mean']
    for m in models.values():m.eval()
    with torch.inference_mode():
        out['MLP 5f'][idx]=models['MLP 5f'](tx(g['windows'],scale,device)).cpu().numpy()*scale['y_std']+scale['y_mean']
        for seq in g['seqs']:
            pred,_=models['LSTM Full'](tx(g['d'][seq],scale,device).reshape(1,-1,1))
            out['LSTM Full'][seq]=pred[0].cpu().numpy()*scale['y_std']+scale['y_mean']
    return out


def scores(y,p):
    e=p-y;den=float(np.sum((y-y.mean())**2))
    return dict(mae_mm=float(np.mean(abs(e))),rmse_mm=float(np.sqrt(np.mean(e**2))),
                r2=float(1-np.sum(e**2)/den) if den>1e-12 else None)


def export_results(output, groups, models, function, scale, history, simulated=False):
    device=next(models['MLP 5f'].parameters()).device
    prefix='SIMULATED / UNTRAINED | ' if simulated else ''
    metrics=[];full_metrics=[];overall=[]
    for split,trials in groups:
        truth=[];pool={label:[] for label in LABELS};full_y=[];full_p=[]
        for g in trials:
            pred=predict(g,models,function,scale,device);idx=g['common'];truth.append(g['y'][idx])
            rows=[]
            for i in range(len(g['d'])):
                rows.append(dict(source_frame=int(g['frame'][i]),time_s=float(g['t'][i]),
                    distance_mm=float(g['d'][i]),actual_aperture_mm=float(g['y'][i]),
                    common_valid=int(i in idx),function_mm=float(pred['Function'][i]),
                    mlp_5f_mm=float(pred['MLP 5f'][i]),lstm_full_mm=float(pred['LSTM Full'][i])))
            write_csv(output/split/'predictions'/f'{g["name"]}.csv',rows)
            full_idx=np.flatnonzero(np.isfinite(pred['LSTM Full']))
            write_csv(output/split/'lstm_full_from_start'/f'{g["name"]}.csv',[rows[i] for i in full_idx])
            full_metrics.append(dict(split=split,trial=g['name'],model='LSTM Full',frames=len(full_idx),**scores(g['y'][full_idx],pred['LSTM Full'][full_idx])))
            full_y.append(g['y'][full_idx]);full_p.append(pred['LSTM Full'][full_idx])
            fig,axs=plt.subplots(1,2,figsize=(13,4.5))
            # Plot each continuous sequence independently: no bridges across resets.
            for j,seq in enumerate(g['seqs']):
                common=seq[np.isin(seq,idx)]
                axs[0].plot(g['t'][seq],g['y'][seq],color='black',lw=1.5,label='Measured' if j==0 else None)
                axs[1].plot(g['d'][seq],g['y'][seq],color='black',lw=1.5,label='Measured' if j==0 else None)
                for label,color in zip(LABELS,COLORS):
                    for ax,x in zip(axs,[g['t'],g['d']]):
                        ax.plot(x[common],pred[label][common],color=color,label=label if j==0 else None)
            for label in LABELS:
                p=pred[label][idx];assert np.isfinite(p).all();pool[label].append(p)
                metrics.append(dict(split=split,trial=g['name'],model=label,frames=len(idx),**scores(g['y'][idx],p)))
            for ax,xlabel in zip(axs,['Actual elapsed time (s)','Index MCP distance (mm)']):
                ax.set_xlabel(xlabel);ax.set_ylabel('Finger aperture (mm)');ax.legend();ax.grid(alpha=.2)
            fig.suptitle(prefix+f'{split}: {g["name"]} | same common target frames');fig.tight_layout()
            dest=output/split/'predictions';dest.mkdir(parents=True,exist_ok=True);fig.savefig(dest/f'{g["name"]}.png',dpi=140);plt.close(fig)
            fig,ax=plt.subplots(figsize=(9,4))
            for j,seq in enumerate(g['seqs']):
                ax.plot(g['t'][seq],g['y'][seq],color='black',label='Measured' if j==0 else None)
                ax.plot(g['t'][seq],pred['LSTM Full'][seq],color=COLORS[2],label='LSTM Full' if j==0 else None)
            ax.set(title=prefix+g['name']+' | full history from first valid frame',xlabel='Actual elapsed time (s)',ylabel='Finger aperture (mm)');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(output/split/'lstm_full_from_start'/f'{g["name"]}.png',dpi=140);plt.close(fig)
        for label in LABELS:
            overall.append(dict(split=split,model=label,frames=sum(map(len,truth)),**scores(np.concatenate(truth),np.concatenate(pool[label]))))
        full_metrics.append(dict(split=split,trial='OVERALL',model='LSTM Full',frames=sum(map(len,full_y)),**scores(np.concatenate(full_y),np.concatenate(full_p))))
    write_csv(output/'per_trial_metrics.csv',metrics);write_csv(output/'overall_metrics.csv',overall);write_csv(output/'lstm_full_metrics.csv',full_metrics);write_csv(output/'training_history.csv',history)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,label in zip(axs,LABELS[1:]):
        h=[r for r in history if r['model']==label]
        if h:ax.plot([r['epoch'] for r in h],[r['train_mae_mm'] for r in h])
        else:ax.text(.5,.5,'No training performed',ha='center',transform=ax.transAxes)
        ax.set(title=prefix+label,xlabel='Epoch',ylabel='Train MAE (mm)');ax.grid(alpha=.2)
    fig.suptitle('Function: one fixed Train-only spline fit; no epoch curve');fig.tight_layout();fig.savefig(output/'training_curves.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(12,4))
    test=[r for r in overall if r['split']=='Test']
    for ax,key,title in zip(axs,['mae_mm','rmse_mm','r2'],['MAE (mm)','RMSE (mm)','R²']):
        bars=ax.bar(LABELS,[r[key] if r[key] is not None else np.nan for r in test],color=COLORS)
        ax.bar_label(bars,fmt='%.3f' if key=='r2' else '%.2f',padding=4)
        ax.margins(y=.18);ax.set_title(title);ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle(prefix+'Test | identical target frames');fig.tight_layout();fig.savefig(output/'test_model_comparison.png',dpi=150);plt.close(fig)


def save_models(output,models,function,scale,simulated=False):
    info=dict(input_column=DISTANCE,target_column=TARGET,units='mm',numeric_scaler=scale,
              scaler_fitted_on='Train only',epochs=0 if simulated else EPOCHS,
              simulated_untrained=simulated,hidden_size=32,dropout=.1,window=WINDOW,
              state_policy='Reset at trial start, missing/invalid input OR target, nonconsecutive frame, nonpositive dt or dt>0.10s',
              physical_processing='No velocity, time normalization, per-trial amplitude normalization, smoothing or interpolation')
    write_json(output/'models'/'function.json',dict(info,architecture='Current-distance cubic smoothing spline',definition=function.definition,epochs=0))
    for label,name in [('MLP 5f','mlp_5f'),('LSTM Full','lstm_full')]:
        f=output/'models'/f'{name}.pth';f.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(info,label=label,input_size=5 if label=='MLP 5f' else 1,output_size=1,parameter_count=sum(p.numel() for p in models[label].parameters()),state_dict={k:v.detach().cpu() for k,v in models[label].state_dict().items()}),f)
    write_json(output/'model_information.json',dict(info,models={
        'Function': 'one scalar current distance -> current aperture; fixed spline lambda=1e-6',
        'MLP 5f': '5 distances oldest-to-current -> Linear(5,32), ReLU, Dropout(.1), Linear(32,32), ReLU, Linear(32,1)',
        'LSTM Full': 'causal unidirectional LSTM(1,32), Dropout(.1), Linear(32,1); full chronological prefix, full BPTT; no state transfer across experiments'},
        fairness='Same common five-consecutive-frame targets for fitting losses and main scoring. LSTM consumes all earlier continuous inputs; separate from-start full metrics.',
        loss='MAE; equal loss weight per experiment; Adam lr=0.001 weight_decay=0.0001',
        selection='Fixed 400 epochs, last epoch only; Function fit once on Train; Test evaluated only after fitting all models.'))


def checks(train,test):
    scale=fit_scaler(train);torch.manual_seed(SEED)
    models={'MLP 5f':MLP(),'LSTM Full':FullLSTM()};models['LSTM Full'].eval()
    before={label:{k:v.clone() for k,v in m.state_dict().items()} for label,m in models.items()}
    # Serialization/prediction exercise for a predefined spline, with NO data fitting.
    definition=dict(kind='smoothing_spline',degree=3,knots=[0,0,0,0,1,1,1,1],coefficients=[0,1,2,3])
    spline=FixedFunction(definition)
    restored=FixedFunction(json.loads(json.dumps(definition)))
    np.testing.assert_allclose(spline.predict(np.array([0.,.5,1.])),restored.predict(np.array([0.,.5,1.])))
    with torch.inference_mode():
        m=models['LSTM Full'];x=torch.arange(12,dtype=torch.float32).reshape(1,12,1)/12
        full,_=m(x);state=None;parts=[]
        for i in range(12):
            p,state=m(x[:,i:i+1],state);parts.append(p)
        torch.testing.assert_close(full,torch.cat(parts,dim=1),atol=1e-6,rtol=1e-5)
        prefix,_=m(x[:,:6]);torch.testing.assert_close(full[:,:6],prefix)
        changed=x.clone();changed[:,6:]+=100
        future,_=m(changed);torch.testing.assert_close(full[:,:6],future[:,:6])
        assert models['MLP 5f'](torch.zeros(3,5)).shape==(3,)
        d=np.arange(9,dtype=float);y=d.copy();t=np.arange(9)*.03;f=np.arange(9);valid=np.ones(9,bool)
        d[2]=np.nan;t[6:]+=.2
        seq=sequence_indices(d,y,t,f,valid);assert [i.tolist() for i in seq]==[[0,1],[3,4,5],[6,7,8]]
        y[4]=np.nan;seq=sequence_indices(d,y,t,f,valid);assert all(4 not in i for i in seq)
        # Each real trial uses its own sequence state, even if inference order changes.
        fn=FixedFunction({'kind':'mock_constant'})
        p1=predict(test[0],models,fn,scale,'cpu')['LSTM Full']
        predict(train[0],models,fn,scale,'cpu')
        p2=predict(test[0],models,fn,scale,'cpu')['LSTM Full']
        np.testing.assert_allclose(p1,p2,equal_nan=True)
    assert all(torch.equal(v,before[label][k]) for label,m in models.items() for k,v in m.state_dict().items())
    return models,FixedFunction({'kind':'mock_constant'}),scale


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--check-data',action='store_true',help='Read-only split/data/model checks; no fits, optimizer, copies or training')
    parser.add_argument('--smoke-export',action='store_true',help='Requires --check-data: export clearly marked UNTRAINED mock results for two trials')
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=parser.parse_args()
    if args.smoke_export and not args.check_data:parser.error('--smoke-export requires --check-data')
    torch.set_num_threads(min(4,torch.get_num_threads()))
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    source=args.source.resolve()
    if not source.is_dir():parser.error(f'Data directory missing: {source}')
    try:train,test,manifest=get_partitions(source,persist=False)
    except (ValueError,OSError,KeyError) as exc:parser.error(str(exc))
    print(f'{len(train)} Train / {len(test)} Test experiments; '+manifest['strategy'],flush=True)
    if args.check_data:
        models,function,scale=checks(train,test)
        report=dict(status='PASS; NO TRAINING OR FUNCTION FIT',epochs_run=0,optimizer_steps=0,
                    train_experiments=len(train),test_experiments=len(test),
                    total_rows=sum(len(g['d']) for g in train+test),common_frames=sum(len(g['common']) for g in train+test),
                    full_history_frames=sum(sum(map(len,g['seqs'])) for g in train+test),
                    excluded_root_directories=[p.name for p in source.iterdir() if p.is_dir() and not re.fullmatch(r'grasp_\d+',p.name) and p.name not in ('Train','Test')],
                    split_preview=manifest,checks=['Calibration, required mm columns and landmark-5 Euclidean geometry','No trial/frame leakage','5-input MLP','1-feature unidirectional LSTM','Full forward = stateful stepwise','Future changes cannot affect past outputs','Missing/invalid/timing resets','No weights changed','Predefined spline serialization without fitting'])
        if args.smoke_export:
            output=source/'Result'/f'CHECK_ONLY_UNTRAINED_{datetime.now():%Y%m%d_%H%M%S_%f}'
            export_results(output,[('Train',train[:1]),('Test',test[:1])],models,function,scale,[],True)
            save_models(output,models,function,scale,True)
            loaded=torch.load(output/'models/mlp_5f.pth',map_location='cpu',weights_only=True);MLP().load_state_dict(loaded['state_dict'])
            loaded=torch.load(output/'models/lstm_full.pth',map_location='cpu',weights_only=True);FullLSTM().load_state_dict(loaded['state_dict'])
            report['checks'].append('Mock CSV/plots/model save-load roundtrip')
            write_json(output/'CHECK_REPORT.json',report)
            (output/'NOT_TRAINED.txt').write_text('SIMULATED EXPORT ONLY. Random neural weights and constant mock function. Metrics are not model performance. No training, function fitting, or optimizer steps occurred. Train/Test copying was not performed.\n',encoding='utf-8')
            print(f'Mock export: {output}',flush=True)
        print(json.dumps({k:v for k,v in report.items() if k!='split_preview'},indent=2));return
    if args.device=='cuda' and not torch.cuda.is_available():parser.error('CUDA unavailable')
    device=torch.device('cuda' if args.device=='cuda' or (args.device=='auto' and torch.cuda.is_available()) else 'cpu')
    train,test,manifest=get_partitions(source,persist=True)
    output=source/'Result'/f'function_mlp5_lstmfull_400ep_{datetime.now():%Y%m%d_%H%M%S_%f}'
    output.mkdir(parents=True);write_json(output/'split_manifest.json',manifest)
    scale=fit_scaler(train);function=fit_function(train,scale)
    models,history=train_networks(train,scale,device)
    save_models(output,models,function,scale)
    export_results(output,[('Train',train),('Test',test)],models,function,scale,history)
    (output/'README.txt').write_text('Function / MLP 5f / LSTM Full. Input: index_mcp_object_distance_mm (Euclidean landmark 5 to object). Output: finger_width_mm.\n400 fixed neural epochs; final epoch saved. Function single Train-only fit; no epoch selection or Test tuning.\nAll plots and metrics use mm (R2 unitless), actual elapsed seconds, original frame IDs. Shared Train-only numeric scaler; no per-trial time/amplitude normalization.\nMain metrics use common valid 5-frame targets for all models. Separate LSTM full-history exports start at first valid frame, resetting on missing/invalid rows and frame/time gaps.\nTrain and Test reports include overall and per-experiment scores. Train MAE history is measured during training with dropout active; final Train metrics use evaluation mode.\nOriginal data preserved; split_manifest.json records whole-experiment partition.\n',encoding='utf-8')
    print(f'Saved last epoch and results: {output}',flush=True)


if __name__=='__main__':main()
