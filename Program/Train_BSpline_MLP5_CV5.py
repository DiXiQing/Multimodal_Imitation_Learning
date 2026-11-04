"""Fixed-knot cubic B-spline regression versus MLP 5f; reference 30mm five-fold split.

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
SOURCE = PROJECT_ROOT/'Data/1101_D/30mm'
DISTANCE = 'index_mcp_object_distance_mm'
TARGET = 'finger_width_mm'
EPOCHS, WINDOW, SEED, MAX_GAP = 200, 5, 20261029, .10
LABELS = ('Function', 'MLP 5f')
COLORS = ('#3485b8', '#e59a32')
SPLINE_LAMBDA = 1e-4  # Predeclared; never chosen using Test.


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



class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5,32), nn.ReLU(), nn.Dropout(.1),
                                 nn.Linear(32,32), nn.ReLU(), nn.Linear(32,1))
    def forward(self, x): return self.net(x).squeeze(-1)


def fit_scaler(train):
    # Feed l/l0 directly so every model input starts at exactly 1.
    # Only output numerical conditioning is fitted on Train; exports invert it to mm.
    x = np.concatenate([g['d'][idx] for g in train for idx in g['seqs']])
    y = np.concatenate([g['y'][g['common']] for g in train])
    return dict(x_mean=0.0, x_std=1.0,
                y_mean=float(y.mean()), y_std=max(float(y.std()),1e-6))


def tx(x, scale, device):
    return torch.tensor((x-scale['x_mean'])/scale['x_std'], dtype=torch.float32, device=device)



REFERENCE = PROJECT_ROOT/'Data/1101_D/Result/30mm_20261103_184459'


class FixedFunction:
    def __init__(self, definition):self.definition=definition
    def predict(self,x):
        s=self.definition
        z=(np.asarray(x)-s['train_x_min'])/s['train_x_range']
        return BSpline(s['knots'],s['coefficients'],3,extrapolate=True)(z)


def fit_function(train,scale):
    # Old 2026-10-17 regression: fixed 6 interior knots, 10 cubic bases,
    # equal trial weights, squared residuals + second-difference penalty.
    x=np.concatenate([g['d'][g['common']] for g in train])
    y=np.concatenate([(g['y'][g['common']]-scale['y_mean'])/scale['y_std'] for g in train])
    weights=np.concatenate([np.full(len(g['common']),1/len(g['common'])) for g in train])
    low=float(x.min());span=float(np.ptp(x))
    if span<=0:raise ValueError('No Train distance variation')
    knots=np.r_[np.zeros(4),np.linspace(0,1,8)[1:-1],np.ones(4)]
    design=BSpline.design_matrix((x-low)/span,knots,3).toarray()
    diff=np.diff(np.eye(design.shape[1]),n=2,axis=0)
    penalty=SPLINE_LAMBDA*(diff.T@diff)*len(train)
    coef=np.linalg.solve(design.T@(weights[:,None]*design)+penalty,design.T@(weights*y))
    return FixedFunction(dict(kind='fixed_knot_cubic_B_spline_regression',degree=3,knots=knots.tolist(),coefficients=coef.tolist(),train_x_min=low,train_x_range=span,lambda_fixed=SPLINE_LAMBDA,basis_count=10,loss='Equal-trial weighted squared residuals + lambda*n_train_trials*sum(second coefficient differences squared)',secondary_x_scaling='Train-only MinMax for B-spline basis construction; physical input remains l/l0'))


def train_mlp(train,scale,device):
    model=MLP().to(device);optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4);rng=random.Random(SEED);history=[]
    for epoch in range(1,EPOCHS+1):
        model.train();total=0.;order=list(range(len(train)));rng.shuffle(order)
        for j in order:
            g=train[j];x=tx(g['windows'],scale,device);y=torch.tensor((g['y'][g['common']]-scale['y_mean'])/scale['y_std'],dtype=torch.float32,device=device)
            optimizer.zero_grad(set_to_none=True);loss=(model(x)-y).abs().mean();loss.backward();optimizer.step();total+=float(loss.detach())*scale['y_std']
        history.append(dict(model='MLP 5f',epoch=epoch,train_mae_mm=total/len(train)))
        if epoch==1 or epoch%25==0:print(f'MLP 5f: {epoch}/{EPOCHS}, Train MAE {total/len(train):.4f} mm',flush=True)
    return model,history


def predict(g,model,function,scale,device):
    idx=g['common'];out={label:np.full(len(g['d']),np.nan) for label in LABELS}
    out['Function'][idx]=function.predict(g['d'][idx])*scale['y_std']+scale['y_mean'];model.eval()
    with torch.inference_mode():out['MLP 5f'][idx]=model(tx(g['windows'],scale,device)).cpu().numpy()*scale['y_std']+scale['y_mean']
    return out

def scores(y,p):
    e=p-y;den=float(np.sum((y-y.mean())**2))
    return dict(mae_mm=float(np.mean(abs(e))),rmse_mm=float(np.sqrt(np.mean(e**2))),
                r2=float(1-np.sum(e**2)/den) if den>1e-12 else None)


def plot_test_comparison(output, metrics, prefix=''):
    # Equal trial weights; SD across trial-level metrics, not frame residuals.
    summary=[]
    fig,axs=plt.subplots(1,3,figsize=(12,4.3))
    for ax,key,title in zip(axs,['mae_mm','rmse_mm','r2'],['MAE (mm)','RMSE (mm)','R²']):
        means=[];stds=[];counts=[]
        for label in LABELS:
            values=[float(r[key]) for r in metrics if r['split']=='Test' and r['model']==label and r.get(key) not in (None,'')]
            values=np.asarray(values);values=values[np.isfinite(values)]
            if not len(values): raise ValueError(f'No finite Test {key} for {label}')
            mean=float(values.mean());sd=float(values.std(ddof=1)) if len(values)>1 else 0.
            means.append(mean);stds.append(sd);counts.append(len(values))
            summary.append(dict(model=label,metric=key,n_trials=len(values),mean=mean,sd=sd,sd_ddof=1))
        ax.bar(LABELS,means,yerr=stds,capsize=5,color=COLORS,error_kw=dict(elinewidth=1.3))
        for i,(mean,sd) in enumerate(zip(means,stds)):
            fmt='.3f' if key=='r2' else '.2f'
            ax.annotate(f'{mean:{fmt}} ± {sd:{fmt}}',(i,mean+sd),xytext=(0,6),textcoords='offset points',ha='center',fontsize=9)
        low=min(0.,min(np.asarray(means)-stds));high=max(np.asarray(means)+stds)
        span=max(high-low,.01);ax.set_ylim(low-.04*span,high+.22*span)
        ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    n=len({r['trial'] for r in metrics if r['split']=='Test'})
    fig.suptitle(prefix+f'Test | trial mean ± SD (n={n}) | identical target frames')
    fig.tight_layout()
    for ext in ('png','svg','pdf'):fig.savefig(output/f'test_model_comparison.{ext}',dpi=150)
    plt.close(fig)
    write_csv(output/'test_model_comparison_mean_sd.csv',summary)



def export(output,groups,model,function,scale,history,simulated=False):
    device=next(model.parameters()).device;metrics=[];overall=[]
    for split,trials in groups:
        truth=[];pool={label:[] for label in LABELS}
        for g in trials:
            pred=predict(g,model,function,scale,device);idx=g['common'];truth.extend(g['y'][idx]);data=[]
            for label in LABELS:
                metrics.append(dict(split=split,trial=g['name'],model=label,frames=len(idx),**scores(g['y'][idx],pred[label][idx])));pool[label].extend(pred[label][idx])
            for i in range(len(g['d'])):data.append(dict(source_frame=int(g['frame'][i]),time_s=float(g['t'][i]),distance_mm=float(g['d_mm'][i]),distance_over_initial=float(g['d'][i]),actual_aperture_mm=float(g['y'][i]),common_valid=int(i in idx),function_mm=float(pred['Function'][i]),mlp_5f_mm=float(pred['MLP 5f'][i])))
            write_csv(output/split/'predictions'/f'{g["name"]}.csv',data)
            fig,ax=plt.subplots(figsize=(6,4))
            for seq in g['seqs']:
                ax.plot(g['t'][seq],g['y'][seq],color='black',lw=1.4)
                for label,col in zip(LABELS,COLORS):ax.plot(g['t'][seq],pred[label][seq],color=col,lw=1.4)
            from matplotlib.lines import Line2D
            ax.legend([Line2D([],[],color=c) for c in ('black',)+COLORS],('Measured',)+LABELS)
            ax.set(xlabel='Time (s)',ylabel='d (mm)',title=('UNTRAINED MOCK | ' if simulated else '')+split+': '+g['name']);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(output/split/'predictions'/f'{g["name"]}.png',dpi=150);plt.close(fig)
        for label in LABELS:overall.append(dict(split=split,model=label,frames=len(truth),**scores(np.array(truth),np.array(pool[label]))))
    write_csv(output/'per_trial_metrics.csv',metrics);write_csv(output/'overall_metrics.csv',overall);write_csv(output/'training_history.csv',history)
    plot_test_comparison(output,metrics,'UNTRAINED MOCK | ' if simulated else '')
    fig,ax=plt.subplots();ax.plot([r['epoch'] for r in history],[r['train_mae_mm'] for r in history]);ax.set(xlabel='Epoch',ylabel='Train MAE (mm)');fig.tight_layout();fig.savefig(output/'training_curve.png');plt.close(fig)
    write_json(output/'models/function.json',dict(definition=function.definition,scaler=scale,input_column=DISTANCE,target_column=TARGET,simulated_untrained=simulated))
    torch.save(dict(state_dict=model.state_dict(),scaler=scale,window=5,dropout=.1,epochs=0 if simulated else EPOCHS,simulated_untrained=simulated),output/'models/mlp_5f.pth')
    return metrics


def reference_folds(source,reference):
    manifest=json.loads((reference/'split_manifest.json').read_text(encoding='utf-8-sig'));folds=[];loaded={}
    for record in manifest['records']:
        name=record['trial'];g=load_trial(source/name)
        if g['csv_sha256']!=record['csv_sha256']:raise ValueError(f'{name}: CSV changed since reference run; not a matched comparison')
        loaded[name]=g
    tests=[]
    for f in manifest['folds']:
        train=[loaded[n] for n in f['train']];test=[loaded[n] for n in f['test']];validate_partitions(train,test);folds.append((train,test));tests.extend(g['name'] for g in test)
    if len(folds)!=5 or len(tests)!=len(loaded) or len(set(tests))!=len(loaded):raise ValueError('Invalid five-fold reference partition')
    return folds,manifest


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def mock_check(train,test):
    scale=fit_scaler(train);model=MLP();before={k:v.clone() for k,v in model.state_dict().items()}
    assert all(g['windows'].shape[1]==5 for g in train+test)
    knots=np.r_[np.zeros(4),np.linspace(0,1,8)[1:-1],np.ones(4)]
    # Arbitrary declared coefficients; NO data fitting during checks.
    fn=FixedFunction(dict(kind='UNTRAINED_MOCK',degree=3,knots=knots.tolist(),coefficients=np.linspace(0,1,10).tolist(),train_x_min=0.,train_x_range=1.))
    x=np.linspace(0,1,20);B=BSpline.design_matrix(x,knots,3).toarray();np.testing.assert_allclose(B.sum(axis=1),1.)
    np.testing.assert_allclose(fn.predict(x),B@np.linspace(0,1,10));np.testing.assert_allclose(fn.predict(x),FixedFunction(json.loads(json.dumps(fn.definition))).predict(x))
    for g in (train[0],test[0]):
        pred=predict(g,model,fn,scale,'cpu');assert all(np.isfinite(pred[label][g['common']]).all() for label in LABELS)
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    return model,fn,scale


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,default=SOURCE);parser.add_argument('--reference',type=Path,default=REFERENCE);parser.add_argument('--check-data',action='store_true');parser.add_argument('--smoke-export',action='store_true');parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');args=parser.parse_args()
    if args.smoke_export and not args.check_data:parser.error('--smoke-export requires --check-data')
    torch.set_num_threads(min(4,torch.get_num_threads()));torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    try:folds,manifest=reference_folds(args.source.resolve(),args.reference.resolve())
    except (OSError,ValueError,KeyError) as e:parser.error(str(e))
    for i,(tr,te) in enumerate(folds,1):print(f'Fold {i}: {len(tr)} Train / {len(te)} Test; reference split reused',flush=True)
    if args.check_data:
        for i,(tr,te) in enumerate(folds):seed_all(SEED+i);mock_check(tr,te)
        if args.smoke_export:
            with tempfile.TemporaryDirectory(prefix='BSPLINE_MLP_MOCK_',dir=Path(__file__).resolve().parent) as temp:
                out=Path(temp);allmetrics=[]
                for i,(tr,te) in enumerate(folds):
                    model,fn,scale=mock_check(tr,te);dest=out/f'fold_{i+1:02d}';allmetrics+=export(dest,[('Train',tr[:1]),('Test',te[:1])],model,fn,scale,[],True);loaded=torch.load(dest/'models/mlp_5f.pth',weights_only=True);MLP().load_state_dict(loaded['state_dict'])
                plot_test_comparison(out,[r for r in allmetrics if r['split']=='Test'],'UNTRAINED MOCK OOF | ')
        print('PASS: exact reference CSV hashes/folds, fixed cubic basis, one-frame function, MLP5, serialization and exports. NO TRAINING OR FUNCTION FIT.');return
    if args.device=='cuda' and not torch.cuda.is_available():parser.error('CUDA unavailable')
    device=torch.device('cuda' if args.device=='cuda' or (args.device=='auto' and torch.cuda.is_available()) else 'cpu')
    out=PROJECT_ROOT/'Data/1101_D/Result'/f'30mm_BSpline_MLP5_{datetime.now():%Y%m%d_%H%M%S}';out.mkdir(parents=True,exist_ok=False);write_json(out/'split_manifest.json',manifest)
    (out/'README.txt').write_text('Two-model matched five-fold rerun of '+str(args.reference)+'\nFunction: fixed-node degree-3 B-spline regression, 6 interior knots, 10 basis coefficients, second-difference penalty lambda=1e-4, equal trial weights. Train-only MinMax basis coordinates after physical l/l0 normalization. MLP5: same 5-32-32-1 architecture, Dropout .1, Adam lr .001, weight_decay .0001, per-trial MAE, 400 final epochs. No LSTM training. No Test tuning or best-fold/epoch selection. Input MCP Euclidean distance mm / trial initial value. Output finger_width_mm read directly; Train-only target standard scaling. Same common valid frames for both models. Exact saved reference split and CSV SHA256 required. Original files untouched; no dataset copies. Reports: out-of-fold per-trial mean ± sample SD across all 200 tests, not across five fold means.\n',encoding='utf-8')
    allmetrics=[]
    for i,(tr,te) in enumerate(folds):
        seed_all(SEED+i);dest=out/f'fold_{i+1:02d}';write_json(dest/'split_manifest.json',manifest['folds'][i]);scale=fit_scaler(tr);print(f'Fold {i+1}/5: fit fixed-knot B-spline on Train only',flush=True);fn=fit_function(tr,scale);model,history=train_mlp(tr,scale,device);allmetrics+=export(dest,[('Train',tr),('Test',te)],model,fn,scale,history)
    oof=[r for r in allmetrics if r['split']=='Test']
    for label in LABELS:assert len({r['trial'] for r in oof if r['model']==label})==manifest['n_trials']
    write_csv(out/'oof_per_trial_metrics.csv',oof);plot_test_comparison(out,oof,'5-fold OOF | ')
    old=list(csv.DictReader((args.reference/'oof_per_trial_metrics.csv').open(encoding='utf-8-sig')))
    old_by={(r['trial'],r['model']):r for r in old};comparison=[]
    for r in oof:
        previous=old_by[(r['trial'],r['model'])]
        comparison.append(dict(trial=r['trial'],model=r['model'],old_mae_mm=previous['mae_mm'],new_mae_mm=r['mae_mm'],mae_change_new_minus_old=r['mae_mm']-float(previous['mae_mm']),old_r2=previous['r2'],new_r2=r['r2']))
    write_csv(out/'comparison_to_reference.csv',comparison);print('Saved:',out,flush=True)


if __name__=='__main__':main()
