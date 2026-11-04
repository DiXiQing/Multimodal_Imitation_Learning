"""Standalone single-frame MLP diagnostic; trial-level 5-fold cross-validation; l/l0; Function / MLP 5f / LSTM Full.

Physical preprocessing: l0 is the first recorded frame distance, not a future
minimum/maximum. Aperture stays in mm. Time stays in actual seconds.

Run --check-data --smoke-export for validation WITHOUT fitting or training.
Run without these flags for 5 independent folds, 400 fixed epochs each, without copying data.
Reference: train_1029_d1_function_mlp_lstm.py (left unchanged).
"""
from __future__ import annotations
import argparse
import tempfile
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

# Default: 20mm dataset. Five folds by complete trial; originals read directly.
# Override with --source if needed.
PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents
                     if (p / 'Data').is_dir()), Path(__file__).resolve().parent)
SOURCE = r'C:\MineApp\Code\Multimodal_Imitation_Learning\Data\1101_D\40mm'
DISTANCE = 'index_mcp_object_distance_mm'
TARGET = 'finger_width_mm'
EPOCHS, WINDOW, SEED, MAX_GAP = 400, 5, 20261029, .10
# WINDOW=5 defines matched training/scoring frames only; model input is ONE current distance.
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
    return sorted(p for p in parent.iterdir() if p.is_dir() and re.fullmatch(r'(?:grasp_\d+|\d{8}_\d{6})', p.name))


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
            windows.append(d[idx[k:k+1]]); indices.append(idx[k])
    if not indices: raise ValueError(f'{folder}: no valid five-frame common targets')
    return dict(name=folder.name, folder=folder, rows=rows, fields=fields, settings=settings,
                d=d, d_mm=distance_mm, initial_distance_mm=initial_distance_mm, y=y, t=t-t[0], unix=t, frame=frame.astype(np.int64), seqs=seqs,
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



def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def make_folds(source):
    trials=[load_trial(p) for p in folders(source)]
    if len(trials)<5:raise ValueError('Need at least 5 complete experiments')
    overrides=[]
    if source.resolve()==(PROJECT_ROOT/'Data/1101_D/20mm').resolve():
        for g in trials:
            overrides.append(dict(trial=g['name'],recorded_size_cm=g['settings'].get('size_cm'),physical_size_mm=20))
            g['settings']=dict(g['settings'],size_cm=2.)
    order=list(range(len(trials)));random.Random(SEED).shuffle(order)
    chunks=[list(map(int,c)) for c in np.array_split(order,5)]
    folds=[]
    for i,chunk in enumerate(chunks):
        held=set(chunk)
        train=[g for j,g in enumerate(trials) if j not in held]
        test=[trials[j] for j in chunk]
        validate_partitions(train,test)
        folds.append((train,test))
    tested=[g['name'] for _,test in folds for g in test]
    assert len(tested)==len(trials)==len(set(tested))
    size_mm=float(trials[0]['settings']['size_cm'])*10
    size_label=f'{size_mm:g}mm'
    manifest=dict(seed=SEED,n_trials=len(trials),n_folds=5,
        strategy='Whole-trial shuffled 5-fold; identical folds for all three models; no file copies',
        object_size_mm=size_mm,label_overrides=overrides,
        folds=[dict(fold=i+1,initialization_seed=SEED+i,train=[g['name'] for g in tr],test=[g['name'] for g in te]) for i,(tr,te) in enumerate(folds)],
        records=[dict(trial=g['name'],source=str(g['folder']),csv_sha256=g['csv_sha256']) for g in trials])
    return folds,manifest,size_label



class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(1,32),nn.ReLU(),nn.Dropout(.1),nn.Linear(32,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x):return self.net(x).squeeze(-1)


def scaler(train):
    y=np.concatenate([g['y'][g['common']] for g in train])
    return dict(x_mean=0.,x_std=1.,y_mean=float(y.mean()),y_std=max(float(y.std()),1e-6))


def scores(y,p):
    e=p-y;den=float(np.sum((y-y.mean())**2))
    return dict(mae_mm=float(np.abs(e).mean()),rmse_mm=float(np.sqrt(np.mean(e**2))),r2=float(1-np.sum(e**2)/den) if den>1e-12 else None)


def fit(train,scale,device):
    model=MLP().to(device);optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4)
    rng=random.Random(SEED);history=[]
    for epoch in range(1,EPOCHS+1):
        model.train();total=0.;order=list(range(len(train)));rng.shuffle(order)
        for j in order:
            g=train[j];idx=g['common'];x=torch.tensor(g['windows'],dtype=torch.float32,device=device)
            y=torch.tensor((g['y'][idx]-scale['y_mean'])/scale['y_std'],dtype=torch.float32,device=device)
            optimizer.zero_grad(set_to_none=True);loss=(model(x)-y).abs().mean();loss.backward();optimizer.step();total+=float(loss.detach())*scale['y_std']
        history.append(dict(epoch=epoch,train_mae_mm=total/len(train)))
        if epoch==1 or epoch%25==0:print(f'MLP 1f {epoch}/{EPOCHS}: Train MAE {total/len(train):.4f} mm',flush=True)
    return model,history


def export(output,groups,model,scale,history,simulated=False):
    model.eval();device=next(model.parameters()).device;metrics=[];initial=[]
    for split,trials in groups:
        for g in trials:
            indices=np.concatenate(g['seqs']);pred=np.full(len(g['d']),np.nan)
            with torch.inference_mode():pred[indices]=model(torch.tensor(g['d'][indices,None],dtype=torch.float32,device=device)).cpu().numpy()*scale['y_std']+scale['y_mean']
            idx=g['common'];metrics.append(dict(split=split,trial=g['name'],model='MLP 1f',frames=len(idx),**scores(g['y'][idx],pred[idx])))
            meta=json.loads((g['folder']/'segment.json').read_text(encoding='utf-8-sig'));onset=meta.get('onset_row');start=meta.get('start_row')
            rest=indices[indices<int(onset)-int(start)] if onset is not None and start is not None else np.array([],dtype=int)
            rest=rest[np.isin(rest,idx)]
            if len(rest):initial.append(dict(split=split,trial=g['name'],frames=len(rest),actual_mean_mm=float(g['y'][rest].mean()),predicted_mean_mm=float(pred[rest].mean()),bias_mm=float((pred[rest]-g['y'][rest]).mean()),**scores(g['y'][rest],pred[rest])))
            write_csv(output/split/'predictions'/f'{g["name"]}.csv',[dict(source_frame=int(g['frame'][i]),time_s=float(g['t'][i]),distance_mm=float(g['d_mm'][i]),distance_over_initial=float(g['d'][i]),actual_aperture_mm=float(g['y'][i]),mlp_1f_mm=float(pred[i]),common_valid=int(i in idx)) for i in range(len(pred))])
            fig,ax=plt.subplots(figsize=(6.5,4));ax.plot(g['t'],g['y'],color='black',label='Measured');ax.plot(g['t'],pred,color='#e59a32',label='MLP 1f');ax.set(xlabel='Time (s)',ylabel='d (mm)',title=('UNTRAINED MOCK | ' if simulated else '')+split+': '+g['name']);ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(output/split/'predictions'/f'{g["name"]}.png',dpi=160);plt.close(fig)
    write_csv(output/'per_trial_metrics.csv',metrics);write_csv(output/'initial_rest_metrics.csv',initial);write_csv(output/'training_history.csv',history)
    output.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(),input_column=DISTANCE,target_column=TARGET,window=1,dropout=.1,scaler=scale,epochs=0 if simulated else EPOCHS,simulated_untrained=simulated),output/'mlp_1f.pth')
    fig,ax=plt.subplots();ax.plot([r['epoch'] for r in history],[r['train_mae_mm'] for r in history]);ax.set(xlabel='Epoch',ylabel='Train MAE (mm)');fig.tight_layout();fig.savefig(output/'training_curve.png');plt.close(fig)
    return metrics,initial


def summarize(output,metrics,initial):
    test=[r for r in metrics if r['split']=='Test'];rest=[r for r in initial if r['split']=='Test'];summary=[]
    for key in ('mae_mm','rmse_mm','r2'):
        v=np.array([float(r[key]) for r in test if r[key] is not None]);v=v[np.isfinite(v)]
        summary.append(dict(metric=key,n_trials=len(v),mean=float(v.mean()),sd=float(v.std(ddof=1))))
    write_csv(output/'oof_per_trial_metrics.csv',test);write_csv(output/'oof_initial_rest_metrics.csv',rest);write_csv(output/'oof_mean_sd.csv',summary)
    fig,axs=plt.subplots(1,3,figsize=(10,3.5))
    for ax,row in zip(axs,summary):
        ax.bar(['MLP 1f'],[row['mean']],yerr=[row['sd']],capsize=5,color='#e59a32');ax.set_title(row['metric']);ax.margins(y=.25)
    fig.tight_layout();fig.savefig(output/'oof_comparison.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,default=SOURCE);parser.add_argument('--check-data',action='store_true');parser.add_argument('--smoke-export',action='store_true');parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');args=parser.parse_args()
    if args.smoke_export and not args.check_data:parser.error('--smoke-export requires --check-data')
    torch.set_num_threads(min(4,torch.get_num_threads()))
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    try:folds,manifest,size=make_folds(args.source.resolve())
    except (OSError,ValueError,KeyError) as e:parser.error(str(e))
    for i,(tr,te) in enumerate(folds,1):print(f'Fold {i}: {len(tr)} Train / {len(te)} Test',flush=True)
    if args.check_data:
        for tr,te in folds:
            assert all(g['windows'].shape==(len(g['common']),1) for g in tr+te)
            seed_all(SEED);model=MLP();scale=scaler(tr)
            assert model(torch.ones(7,1)).shape==(7,)
        if args.smoke_export:
            with tempfile.TemporaryDirectory(prefix='MLP1_UNTRAINED_',dir=Path(__file__).resolve().parent) as temp:
                out=Path(temp);metrics=[];initial=[]
                for i,(tr,te) in enumerate(folds):
                    seed_all(SEED+i);model=MLP();scale=scaler(tr);before={k:v.clone() for k,v in model.state_dict().items()}
                    dest=out/f'fold_{i+1:02d}';a,b=export(dest,[('Train',tr[:1]),('Test',te[:1])],model,scale,[],True);metrics+=a;initial+=b
                    state=torch.load(dest/'mlp_1f.pth',weights_only=True);MLP().load_state_dict(state['state_dict']);assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
                summarize(out,metrics,initial)
        print('PASS: 1-feature MLP, trial-level folds, first-point normalization, mock save/load/export. NO TRAINING; optimizer steps=0.');return
    if args.device=='cuda' and not torch.cuda.is_available():parser.error('CUDA unavailable')
    device=torch.device('cuda' if args.device=='cuda' or (args.device=='auto' and torch.cuda.is_available()) else 'cpu')
    out=PROJECT_ROOT/'Data/1101_D/Result'/f'{size}_MLP1_{datetime.now():%Y%m%d_%H%M%S}';out.mkdir(parents=True,exist_ok=False);write_json(out/'split_manifest.json',manifest)
    (out/'README.txt').write_text('Single current normalized MCP distance -> MLP(1,32,32,1), ReLU, Dropout(0.1). Output mm. 5-fold trial CV, 400 final epochs per fold; no Test tuning. Same fixed seed and fold construction as Train_CV5 on the SAME active dataset. Only current distance is input. Matched training/scoring frames exclude first four frames of each sequence, exactly as MLP5; predictions exported for ALL valid frames. No padding. Initial-rest diagnostics use those common frames before annotated onset. Output scaling fitted only on fold Train. Original data not copied or edited. Changing the dataset changes fold membership; compare by saved lists, not only seed.',encoding='utf-8')
    metrics=[];initial=[]
    for i,(tr,te) in enumerate(folds):
        seed_all(SEED+i);scale=scaler(tr);print(f'Fold {i+1}/5: MLP 1f on {device}',flush=True);model,history=fit(tr,scale,device);dest=out/f'fold_{i+1:02d}';write_json(dest/'split_manifest.json',manifest['folds'][i]);a,b=export(dest,[('Train',tr),('Test',te)],model,scale,history);metrics+=a;initial+=b
    summarize(out,metrics,initial);print('Saved:',out,flush=True)


if __name__=='__main__':main()
