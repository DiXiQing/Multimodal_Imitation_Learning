"""Combine saved predictions; never fit or select models."""
from pathlib import Path
import csv
import json
import shutil
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] / 'Data/1101_D/Result'
A = ROOT / '30mm_20261103_184459'
B = ROOT / '30mm_BSpline_MLP5_20261103_194139'
LABELS = {'Function': 'function_mm', 'MLP 5f': 'mlp_5f_mm', 'LSTM Full': 'lstm_full_mm'}

def read(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def scores(y, p):
    y, p = np.asarray(y), np.asarray(p)
    assert np.isfinite(y).all() and np.isfinite(p).all()
    e = p-y
    return dict(mae_mm=float(np.abs(e).mean()), rmse_mm=float(np.sqrt((e*e).mean())),
                r2=float(1-(e*e).sum()/((y-y.mean())**2).sum()))

def report(out, rows):
    summary=[]
    fig, axes=plt.subplots(1,3,figsize=(12,4.3))
    for ax, key, title in zip(axes,['mae_mm','rmse_mm','r2'],['MAE (mm)','RMSE (mm)','R²']):
        means=[]; sds=[]
        for label in LABELS:
            v=np.array([r[key] for r in rows if r['model']==label],dtype=float)
            mean,sd=float(v.mean()),float(v.std(ddof=1))
            means.append(mean);sds.append(sd)
            summary.append(dict(model=label,metric=key,n_trials=len(v),mean=mean,sd=sd,sd_ddof=1))
        ax.bar(list(LABELS),means,yerr=sds,capsize=5,color=['#4C78A8','#F58518','#54A24B'])
        for i,(m,s) in enumerate(zip(means,sds)):
            ax.annotate(f'{m:.3f} ± {s:.3f}',(i,m+s),xytext=(0,6),textcoords='offset points',ha='center',fontsize=9)
        lo=min(0,min(np.array(means)-sds)); hi=max(np.array(means)+sds); span=max(hi-lo,.01)
        ax.set_ylim(lo-.04*span,hi+.22*span);ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle(f'Test | trial mean ± SD (n={len(rows)//3}) | identical target frames')
    fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(out/f'test_model_comparison.{ext}',dpi=160)
    plt.close(fig);write(out/'test_model_comparison_mean_sd.csv',summary)

def main():
    # Validate every prediction before creating the destination.
    checked=[]
    for fold in range(1,6):
        name=f'fold_{fold:02d}'
        ma=json.loads((A/name/'split_manifest.json').read_text(encoding='utf-8'))
        mb=json.loads((B/name/'split_manifest.json').read_text(encoding='utf-8'))
        for key in ['train','test']:assert ma[key]==mb[key],(name,key)
        for split in ['Train','Test']:
            pa=A/name/split/'predictions';pb=B/name/split/'predictions'
            assert {f.name for f in pa.glob('*.csv')}=={f.name for f in pb.glob('*.csv')}
            for f in sorted(pa.glob('*.csv')):
                ra,rb=read(f),read(pb/f.name)
                assert len(ra)==len(rb)
                common=set(ra[0]) & set(rb[0]) - {'function_mm','mlp_5f_mm'}
                for x,y in zip(ra,rb):
                    for key in common:
                        np.testing.assert_allclose(float(x[key]),float(y[key]),rtol=0,atol=1e-10,equal_nan=True,err_msg=f'{name}/{split}/{f.name}/{key}')
                    x['function_mm']=y['function_mm'];x['mlp_5f_mm']=y['mlp_5f_mm']
                checked.append((fold,split,f.stem,ra))
    out=ROOT/f'30mm_Combined_BSpline_MLP5_LSTMFull_{datetime.now():%Y%m%d_%H%M%S}'
    out.mkdir(exist_ok=False)
    allmetrics=[];overall=[];oofy=[];oofp={k:[] for k in LABELS}
    for fold in range(1,6):
        name=f'fold_{fold:02d}';dest=out/name;dest.mkdir()
        shutil.copy2(A/name/'split_manifest.json',dest/'split_manifest.json')
        (dest/'models').mkdir()
        for src,filename in [(B,'function.json'),(B,'mlp_5f.pth'),(A,'lstm_full.pth')]:
            shutil.copy2(src/name/'models'/filename,dest/'models'/filename)
        for filename in ['initial_distances.csv','lstm_full_metrics.csv']:
            shutil.copy2(A/name/filename,dest/filename)
        metrics=[];pooled=[]
        for split in ['Train','Test']:
            ys=[];ps={k:[] for k in LABELS}
            for n,s,trial,rows in checked:
                if n!=fold or s!=split:continue
                write(dest/split/'predictions'/f'{trial}.csv',rows)
                valid=[r for r in rows if int(r['common_valid'])]
                y=[float(r['actual_aperture_mm']) for r in valid];ys.extend(y)
                for label,col in LABELS.items():
                    p=[float(r[col]) for r in valid];ps[label].extend(p)
                    metrics.append(dict(fold=fold,split=split,trial=trial,model=label,frames=len(y),**scores(y,p)))
            for label in LABELS:pooled.append(dict(fold=fold,split=split,model=label,frames=len(ys),**scores(ys,ps[label])))
            if split=='Test':
                oofy.extend(ys)
                for label in LABELS:oofp[label].extend(ps[label])
        write(dest/'per_trial_metrics.csv',metrics);write(dest/'overall_metrics.csv',pooled)
        report(dest,[r for r in metrics if r['split']=='Test'])
        allmetrics.extend(metrics);overall.extend(pooled)
    tests=[r for r in allmetrics if r['split']=='Test']
    assert len(tests)==600 and len({r['trial'] for r in tests})==200
    # Recomputed scores must reproduce the selected saved metrics.
    for src,labels in [(A,{'LSTM Full'}),(B,{'Function','MLP 5f'})]:
        expected={(r['trial'],r['model']):r for r in read(src/'oof_per_trial_metrics.csv') if r['model'] in labels}
        for r in tests:
            if r['model'] in labels:
                for key in ['mae_mm','rmse_mm','r2']:
                    np.testing.assert_allclose(r[key],float(expected[r['trial'],r['model']][key]),rtol=1e-10,atol=1e-10)
    write(out/'oof_per_trial_metrics.csv',tests)
    write(out/'fold_test_overall_metrics.csv',[r for r in overall if r['split']=='Test'])
    write(out/'oof_frame_pooled_metrics.csv',[dict(model=k,frames=len(oofy),**scores(oofy,oofp[k])) for k in LABELS])
    shutil.copy2(A/'oof_lstm_full_from_start_metrics.csv',out/'oof_lstm_full_from_start_metrics.csv')
    report(out,tests)
    provenance=dict(sources={k:str(B if k!='LSTM Full' else A) for k in LABELS},validation=dict(prediction_files=len(checked),test_trials=200,test_frames=len(oofy),matched_splits=True,matched_frames_inputs_targets_masks=True,recomputed_scores_match_sources=True),method='Existing predictions combined without retraining. Main scores use common_valid frames; equal-trial mean and sample SD (ddof=1).')
    (out/'source_manifest.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'README.txt').write_text('Combined comparison: Function (B-spline) and MLP 5f from '+str(B)+'; LSTM Full from '+str(A)+'.\nNo retraining or model selection. Sources untouched.\nFive matched folds; 200 held-out trials; all shared frames, inputs, targets and valid masks verified. Main comparison uses common_valid targets. Trial mean and sample SD (ddof=1), not SD of fold means. Separate frame-pooled scores and LSTM-from-start scores.\nEach fold contains combined Train/Test prediction CSVs, selected model files and recomputed metrics. Models preserve their own embedded preprocessing. See source_manifest.json for provenance.\n',encoding='utf-8')
    print(out)
    print(json.dumps(provenance['validation']))

if __name__=='__main__':main()
