"""49-trial offline, trial-level 5-fold CV: Function / MLP 5f / LSTM Full.
Run normally to train 400 epochs per neural model per fold.
--check-data --smoke-export performs checks only, with temporary untrained exports.
"""
from pathlib import Path
from datetime import datetime
import argparse,csv,json,random,tempfile,shutil,re
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_1031_d_mixed_l0_function_mlp_lstm as core

SOURCE=Path(__file__).resolve().parents[2]/'Data/1101_D/20mm'
SEED=20261029
N_FOLDS=5

def make_folds(trials):
    if len(trials)<N_FOLDS:raise ValueError('Need at least five complete trials')
    if len({g['csv_sha256'] for g in trials})!=len(trials):raise ValueError('Duplicate trial CSVs')
    order=list(range(len(trials)));random.Random(SEED).shuffle(order)
    test_groups=np.array_split(order,N_FOLDS)
    folds=[]
    for n,indices in enumerate(test_groups,1):
        ids=set(indices.tolist())
        train=[g for i,g in enumerate(trials) if i not in ids]
        test=[g for i,g in enumerate(trials) if i in ids]
        core.validate_partitions(train,test)
        folds.append((n,train,test))
    names=[g['name'] for _,_,test in folds for g in test]
    assert len(names)==len(trials) and len(set(names))==len(trials)
    return folds

def summarize(rows, output, simulated=False):
    core.write_csv(output/'oof_per_trial_metrics.csv',rows)
    summary=[]
    for label in core.LABELS:
        subset=[r for r in rows if r['model']==label]
        assert len(subset)==len(set(r['trial'] for r in subset))
        result=dict(model=label,trials=len(subset))
        for metric in ('mae_mm','rmse_mm','r2'):
            values=np.array([float(r[metric]) for r in subset if r.get(metric) not in (None,'')])
            result[metric+'_mean']=float(values.mean()) if len(values) else None
            result[metric+'_sd']=float(values.std(ddof=1)) if len(values)>1 else None
            result[metric+'_valid_trials']=len(values)
        summary.append(result)
    core.write_csv(output/'oof_mean_sd.csv',summary)
    fig,axs=plt.subplots(1,3,figsize=(12,4.5))
    for ax,metric,title in zip(axs,('mae_mm','rmse_mm','r2'),('MAE (mm)','RMSE (mm)','R²')):
        mean=[r[metric+'_mean'] for r in summary];sd=[r[metric+'_sd'] for r in summary]
        bars=ax.bar(core.LABELS,mean,yerr=sd,capsize=5,color=core.COLORS)
        for bar,m,s in zip(bars,mean,sd):
            if m is not None and s is not None:
                ax.text(bar.get_x()+bar.get_width()/2,m+s,f'{m:.3f} ± {s:.3f}' if metric=='r2' else f'{m:.2f} ± {s:.2f}',ha='center',va='bottom',fontsize=8)
        ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.margins(y=.23)
    fig.suptitle(('UNTRAINED CHECK ONLY | ' if simulated else '')+'5-fold out-of-fold evaluation | trial mean ± SD')
    fig.tight_layout();fig.savefig(output/'test_model_comparison_mean_sd.png',dpi=180);plt.close(fig)
    return summary

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--check-data',action='store_true')
    parser.add_argument('--smoke-export',action='store_true',help='Requires --check-data; temporary UNTRAINED exports, removed after checks')
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args=parser.parse_args()
    if args.smoke_export and not args.check_data:parser.error('--smoke-export requires --check-data')
    source=args.source.resolve()
    if not source.is_dir():parser.error(f'Data directory missing: {source}')
    torch.set_num_threads(min(4,torch.get_num_threads()))
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    try:
        trial_folders=sorted(p for p in source.iterdir() if p.is_dir()
                             and re.fullmatch(r'(?:grasp_\d+|\d{8}_\d{6})',p.name))
        trials=[core.load_trial(p) for p in trial_folders]
        folds=make_folds(trials)
    except (ValueError,OSError,KeyError) as exc:parser.error(str(exc))
    print(f'{len(trials)} trials | 5-fold trial-level CV | split seed {SEED}',flush=True)
    for n,train,test in folds:print(f'Fold {n}: {len(train)} Train / {len(test)} Test',flush=True)
    print('Input: index_mcp_object_distance_mm / first-frame distance. Output: finger_width_mm.',flush=True)
    hashes={g['folder']/'camera.csv':g['csv_sha256'] for g in trials}
    manifest=dict(seed=SEED,strategy='5-fold whole-trial random split; identical folds for all models; each trial tested exactly once',trials=len(trials),folds=[dict(fold=n,training_seed=SEED+n,train=[g['name'] for g in train],test=[g['name'] for g in test]) for n,train,test in folds],records=[dict(trial=g['name'],source=str(g['folder']),csv_sha256=g['csv_sha256']) for g in trials])
    if args.check_data:
        mock_rows=[]
        with tempfile.TemporaryDirectory(prefix='check_cv5_',dir=Path(__file__).resolve().parent) as temp:
            temp=Path(temp)
            for n,train,test in folds:
                models,function,scale=core.checks(train,test)
                # Verify numerical conditioning against training targets only.
                train_targets=np.concatenate([g['y'][g['common']] for g in train])
                assert np.isclose(scale['y_mean'],train_targets.mean())
                assert np.isclose(scale['y_std'],max(train_targets.std(),1e-6))
                assert scale['x_mean']==0.0 and scale['x_std']==1.0
                for g in test:
                    pred=core.predict(g,models,function,scale,'cpu')
                    for label in core.LABELS:
                        idx=g['common'];assert np.isfinite(pred[label][idx]).all()
                        mock_rows.append(dict(fold=n,split='Test',trial=g['name'],model=label,frames=len(idx),**core.scores(g['y'][idx],pred[label][idx])))
                if args.smoke_export and n==1:
                    core.export_results(temp/'fold_01',[('Train',train[:1]),('Test',test[:1])],models,function,scale,[],True)
                    core.save_models(temp/'fold_01',models,function,scale,True)
                    for label,filename,cls in [('MLP 5f','mlp_5f.pth',core.MLP),('LSTM Full','lstm_full.pth',core.FullLSTM)]:
                        checkpoint=torch.load(temp/'fold_01/models'/filename,map_location='cpu',weights_only=True)
                        restored=cls();restored.load_state_dict(checkpoint['state_dict'])
                        for k,v in models[label].state_dict().items():assert torch.equal(v,restored.state_dict()[k])
            if args.smoke_export:summarize(mock_rows,temp,True)
            assert len(mock_rows)==len(trials)*3
        assert all(core.file_hash(p)==digest for p,digest in hashes.items())
        print(json.dumps(dict(status='PASS: no training, spline fitting, optimizer steps or persistent result exports',trials=len(trials),total_frames=sum(len(g['d']) for g in trials),common_frames=sum(len(g['common']) for g in trials),checks=['Calibration and landmark-5 Euclidean distance','First-frame l/l0 == 1','Five disjoint held-out folds cover every trial once','No recorded-frame leakage','Train-only output scaling','MLP five consecutive real frames; no initial padding','LSTM causal full history; per-trial and gap resets','Each held-out trial scored once per model','Mock export/model roundtrip in temporary folder' if args.smoke_export else 'No export requested','Source CSV hashes unchanged']),indent=2))
        return
    if args.device=='cuda' and not torch.cuda.is_available():parser.error('CUDA unavailable')
    device=torch.device('cuda' if args.device=='cuda' or (args.device=='auto' and torch.cuda.is_available()) else 'cpu')
    output=source/'Result'/f'cv5_l0_function_mlp5_lstmfull_400ep_{datetime.now():%Y%m%d_%H%M%S_%f}'
    output.mkdir(parents=True)
    core.write_json(output/'fold_manifest.json',manifest)
    core.write_csv(output/'initial_distances.csv',[dict(trial=g['name'],initial_distance_mm=g['initial_distance_mm']) for g in trials])
    all_metrics=[];full_metrics=[]
    for n,train,test in folds:
        seed_all(SEED+n);core.SEED=SEED+n
        dest=output/f'fold_{n:02d}';dest.mkdir()
        print(f'\nFold {n}/5 | {len(train)} Train, {len(test)} Test | device {device} | 400 epochs per neural model',flush=True)
        scale=core.fit_scaler(train)
        print('Fitting Function on Train only...',flush=True)
        function=core.fit_function(train,scale)
        models,history=core.train_networks(train,scale,device)
        core.save_models(dest,models,function,scale)
        core.export_results(dest,[('Train',train),('Test',test)],models,function,scale,history)
        for filename,target in [('per_trial_metrics.csv',all_metrics),('lstm_full_metrics.csv',full_metrics)]:
            with (dest/filename).open(encoding='utf-8-sig',newline='') as f:
                target.extend(dict(fold=n,**r) for r in csv.DictReader(f) if r['split']=='Test' and r['trial']!='OVERALL')
        # Gather only held-out predictions, each trial exactly once.
        for g in test:
            for sub in ('predictions','lstm_full_from_start'):
                target=output/'OOF'/sub;target.mkdir(parents=True,exist_ok=True)
                for ext in ('csv','png'):shutil.copy2(dest/'Test'/sub/f'{g["name"]}.{ext}',target/f'{g["name"]}.{ext}')
        core.write_csv(output/'oof_per_trial_metrics.csv',all_metrics)
        del models
    assert len(all_metrics)==3*len(trials)
    summarize(all_metrics,output)
    core.write_csv(output/'oof_lstm_full_from_start_metrics.csv',full_metrics)
    assert all(core.file_hash(p)==digest for p,digest in hashes.items())
    (output/'README.txt').write_text('5-fold trial-level cross-validation. Each complete trial is held out once, using the same fold for Function, MLP 5f and LSTM Full. Main report: oof_mean_sd.csv and test_model_comparison_mean_sd.png (per-trial MAE/RMSE/R2 mean and sample SD across held-out trials, not SD across five fold averages).\nEach fold trains fresh models, fits output numeric scale on Train only, and fits Function on Train only with fixed lambda. Neural models train exactly 400 epochs and save the last epoch; Test never selects parameters or epochs.\nInput: index_mcp_object_distance_mm / its first recorded value, dimensionless l/l0; input scaler identity. Target: finger_width_mm. No time normalization, aperture amplitude normalization, smoothing, velocity or future inputs.\nMLP waits for five real consecutive inputs. LSTM uses the full continuous causal history, reset at trial start or invalid/frame/time gaps; initial distance baseline retained across gaps. All main losses and metrics use identical valid frames. Separate LSTM-from-start predictions include early valid frames.\nOOF contains held-out predictions only. Each fold contains Train/Test diagnostics, training curves and three final models. Train results are not the reported cross-validation score. There is no single final model trained on all trials.\nTrials are read in place; backup, review_segments, derived folders and existing Train/Test directories are excluded. Original data unchanged. Similarity selection preceded this evaluation; results describe the selected trials.\n',encoding='utf-8')
    print(f'Saved 5-fold models and offline evaluation: {output}',flush=True)

if __name__=='__main__':main()
