"""Trial-grouped regression: 17 development trials, 5 held-out test trials."""
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy.interpolate import BSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from trial_paths import grasp_folders
from median_regression import fit_median_spline

# Optional absolute path, e.g. r'C:\MineApp\Code\...\grasp_segments'.
# Leave empty to paste the absolute path when the program starts.
SOURCE=r'C:\MineApp\Code\Multimodal_Imitation_Learning\New\Data\2'
SEED=20261016
# Defaults for running directly from an IDE; CLI flags override these values.
TRAIN_COUNT=40
TEST_COUNT=20
MODEL_MODE='mean'  # median: middle-value regression; mean: previous models; auto: compare all
SELECTION_METRIC='mae'  # Optimize median models using validation MAE by default.


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def fit(groups,kind,param):
    x=np.concatenate([g['x'] for g in groups]);y=np.concatenate([g['y'] for g in groups])
    weights=np.concatenate([np.full(len(g['x']),1/len(g['x'])) for g in groups])
    low=float(x.min());scale=float(x.max()-low)
    if not np.isfinite(scale) or scale<=0:
        raise ValueError('Training distances must span more than one value.')
    z=(x-low)/scale
    model=dict(kind=kind,param=param,x_min=low,x_max=float(x.max()),scale=scale)
    if kind=='median_spline':
        model.update(fit_median_spline(z,y,weights,param))
        return model
    if kind=='mean':
        model['coefficients']=[float(np.average(y,weights=weights))];return model
    if kind=='polynomial':
        design=np.polynomial.polynomial.polyvander(z,int(param));penalty=np.zeros((int(param)+1,int(param)+1))
    else:
        knots=np.r_[np.zeros(4),np.linspace(0,1,8)[1:-1],np.ones(4)]
        design=BSpline.design_matrix(z,knots,3).toarray()
        difference=np.diff(np.eye(design.shape[1]),n=2,axis=0)
        penalty=float(param)*(difference.T@difference)*len(groups)
        model['knots']=knots.tolist();model['degree']=3
    root=np.sqrt(weights)
    # Augmented least squares avoids normal-equation conditioning for polynomials.
    if kind=='polynomial':
        coef=np.linalg.lstsq(design*root[:,None],y*root,rcond=None)[0]
    else:
        coef=np.linalg.solve(design.T@(weights[:,None]*design)+penalty,design.T@(weights*y))
    model['coefficients']=coef.tolist()
    return model


def predict(model,x):
    z=(np.asarray(x)-model['x_min'])/model['scale']
    if model['kind']=='mean':return np.full_like(z,model['coefficients'][0])
    if model['kind']=='polynomial':return np.polynomial.polynomial.polyval(z,model['coefficients'])
    return BSpline(model['knots'],model['coefficients'],model['degree'],extrapolate=True)(z)


def metrics(y,p):
    error=p-y
    return dict(mae=float(np.mean(abs(error))),rmse=float(np.sqrt(np.mean(error**2))),
                bias=float(np.mean(error)),r2=float(1-np.sum(error**2)/np.sum((y-y.mean())**2)))


def plot_test_metrics(reports, overall, output):
    """One chart with separate axes for pixel errors and dimensionless R-squared."""
    chart_rows=[dict(label=f'Test {i:02d}',trial=r['trial'],frames=r['frames'],
                     mae=r['mae'],rmse=r['rmse'],r2=r['r2']) for i,r in enumerate(reports,1)]
    chart_rows.append(dict(label='Overall',trial='All test frames pooled',
                           frames=sum(r['frames'] for r in reports),
                           **{key:overall[key] for key in ('mae','rmse','r2')}))
    write_csv(output/'test_metrics_chart_data.csv',chart_rows)
    fig,axes=plt.subplots(1,3,figsize=(13,max(4.2,.38*len(chart_rows)+2)))
    positions=np.arange(len(chart_rows))
    colors=['#3485b8']*len(reports)+['#e59a32']
    for ax,key,title in zip(axes,['mae','rmse','r2'],
                            ['MAE (px) | lower is better','RMSE (px) | lower is better',
                             'R\u00b2 | higher is better']):
        values=np.array([r[key] for r in chart_rows],dtype=float)
        finite=np.isfinite(values)
        ax.barh(positions[finite],values[finite],color=np.asarray(colors)[finite],height=.65)
        ax.set_yticks(positions,[r['label'] for r in chart_rows]);ax.invert_yaxis()
        ax.set_title(title,fontsize=11);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        low=min(0,float(np.min(values[finite]))) if finite.any() else 0
        high=max(1 if key=='r2' else 0,float(np.max(values[finite]))) if finite.any() else 1
        span=max(high-low,.1)
        ax.set_xlim(low-.12*span if low<0 else 0,high+.25*span)
        ax.axvline(0,color='#777777',lw=.6)
        if key=='r2':ax.axvline(1,color='#777777',ls=':',lw=.8)
        for i,value in enumerate(values):
            label=f'{value:.3f}' if key=='r2' else f'{value:.2f}'
            if np.isfinite(value):
                ax.text(value+(.02*span if value>=0 else -.02*span),i,label,
                        ha='left' if value>=0 else 'right',va='center',fontsize=9)
            else:ax.text(.02,i,'N/A',va='center',transform=ax.get_yaxis_transform(),fontsize=9)
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    fig.suptitle('Held-out test metrics | per trial and overall',fontsize=15)
    fig.text(.5,.025,'Blue: individual test trials. Orange: all test frames pooled. '
             'Trial-name mapping: test_metrics_chart_data.csv. R\u00b2 is not accuracy.',
             ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.93))
    fig.savefig(output/'test_metrics_chart.png',dpi=180)
    fig.savefig(output/'test_metrics_chart.pdf')
    plt.close(fig)


def main():
    run_started=time.perf_counter()
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description='按完整抓握划分训练/测试，训练组内部验证选择回归模型。')
    parser.add_argument('--source',default=SOURCE,help='grasp_segments 的绝对路径；不填写则启动后输入')
    parser.add_argument('--train-count',type=int,default=TRAIN_COUNT,help=f'训练试次数，默认 {TRAIN_COUNT}')
    parser.add_argument('--test-count',type=int,default=TEST_COUNT,help='测试试次数，默认 5')
    parser.add_argument('--seed',type=int,default=SEED,help='随机种子，默认 20261016')
    parser.add_argument('--model-mode',choices=['median','mean','auto'],default=MODEL_MODE,
                        help='median: 中位数回归（默认）；mean: 原模型；auto: 内部验证比较后选择')
    parser.add_argument('--selection-metric',choices=['mae','rmse'],default=SELECTION_METRIC,
                        help='训练内部验证的选模指标，默认 mae')
    parser.add_argument('--output','--output-dir',type=Path,help='结果目录；默认 数据目录/Result/运行时间')
    args=parser.parse_args()
    source_text=str(args.source).strip()
    if not source_text:
        try:
            source_text=input('请输入训练数据文件夹的绝对路径（包含 grasp_001 等子目录）：\n> ').strip()
        except (EOFError,KeyboardInterrupt):
            parser.error('未输入数据目录。也可以使用 --source 指定绝对路径。')
    args.source=Path(source_text.strip('\"\''))
    if not source_text or not args.source.is_absolute():
        parser.error('请输入完整的绝对路径，例如 C:\\MineApp\\Code\\...\\grasp_segments。')
    if args.train_count<3 or args.test_count<1:
        parser.error('训练次数至少为 3，测试次数至少为 1。')
    if not args.source.is_dir():
        parser.error(f'数据目录不存在：{args.source}')
    groups=[]
    for folder in grasp_folders(args.source):
        with (folder/'camera.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        rows=[r for r in rows if r['valid']=='1' and r['hand_object_distance'] and r['finger_width']]
        x=np.array([float(r['hand_object_distance']) for r in rows]);y=np.array([float(r['finger_width']) for r in rows])
        if len(x)<2:
            parser.error(f'{folder.name} 有效帧不足 2 帧。')
        assert np.isfinite(x).all() and np.isfinite(y).all()
        groups.append(dict(id=folder.name,x=x,y=y,rows=rows))
    if args.train_count+args.test_count!=len(groups):
        parser.error(f'发现 {len(groups)} 次抓握，train-count + test-count 必须等于 {len(groups)}。')
    OUT=args.output or args.source/'Result'/f'regression_{args.train_count}train_{args.test_count}test_{datetime.now():%Y%m%d_%H%M%S_%f}'
    if OUT.exists():
        parser.error(f'结果目录已存在，请换一个目录，避免覆盖：{OUT}')
    OUT.mkdir(parents=True)
    (OUT/'models').mkdir()
    (OUT/'test_predictions').mkdir()
    log_lines=[]
    def log(message=''):
        print(message,flush=True)
        log_lines.append(message)
        (OUT/'training_log.txt').write_text('\n'.join(log_lines)+'\n',encoding='utf-8')
    order=np.random.default_rng(args.seed).permutation(len(groups))
    test_indices=set(order[:args.test_count].tolist())
    train=[g for i,g in enumerate(groups) if i not in test_indices]
    test=[g for i,g in enumerate(groups) if i in test_indices]
    assert not ({g['id'] for g in train}&{g['id'] for g in test})
    log('=== Camera-only grasp regression ===')
    log(f'Data: {args.source.resolve()}')
    log('Input: hand_object_distance (px) -> Output: finger_width (px)')
    log(f'Training: {len(train)} trials / {sum(len(g["x"]) for g in train)} frames')
    log('  '+', '.join(g['id'] for g in train))
    log(f'Test: {len(test)} trials / {sum(len(g["x"]) for g in test)} frames')
    log('  '+', '.join(g['id'] for g in test))
    log(f'Seed: {args.seed} | Validation: leave-one-trial-out')
    log(f'Model mode: {args.model_mode} | Selection criterion: validation {args.selection_metric.upper()}')
    log(f'Results: {OUT.resolve()}')
    split=dict(seed=args.seed,train_trials=[g['id'] for g in train],test_trials=[g['id'] for g in test],
        strategy=f'Seeded random complete-trial split; first {len(test)} shuffled trial indices are held out.',
        validation=f'Leave-one-trial-out on {len(train)} training trials; choose lowest mean validation {args.selection_metric.upper()} within {args.model_mode} mode.',
        model_mode=args.model_mode,selection_metric=args.selection_metric,
        weighting='Equal total weight per training trial; validation score is mean of trial RMSE.',
        data='All raw valid segment frames, including pre-movement rest and grasp hold; no smoothing or clipping.')
    (OUT/'split.json').write_text(json.dumps(split,indent=2),encoding='utf-8')
    write_csv(OUT/'data_split.csv',[dict(trial=g['id'],split=label,frames=len(g['x']))
        for label,items in [('train',train),('test',test)] for g in items])
    candidates=([('polynomial',d) for d in (1,2,3,4)]
                +[('spline',s) for s in (.0001,.001,.01,.1,1,10)]
                +[('median_spline',s) for s in (0,.001,.01,.1)])
    log(f'\n[1/3] Fit and validate {len(candidates)} candidates ({len(candidates)*len(train)} fits)')
    results=[];folds=[]
    fitting_started=time.perf_counter()
    for candidate_index,(kind,param) in enumerate(candidates,1):
        candidate_started=time.perf_counter()
        scores=[]
        for i,validation in enumerate(train):
            model=fit(train[:i]+train[i+1:],kind,param)
            score=metrics(validation['y'],predict(model,validation['x']))
            scores.append(score)
            folds.append(dict(kind=kind,param=param,validation_trial=validation['id'],**score))
        results.append(dict(kind=kind,param=param,cv_rmse=float(np.mean([s['rmse'] for s in scores])),
            cv_mae=float(np.mean([s['mae'] for s in scores])),cv_rmse_sd=float(np.std([s['rmse'] for s in scores]))))
        log(f'  [{candidate_index:02d}/{len(candidates)}] {kind:<10} param={param:<7g} '
            f'CV RMSE={results[-1]["cv_rmse"]:6.2f}px | MAE={results[-1]["cv_mae"]:6.2f}px '
            f'| {time.perf_counter()-candidate_started:.2f}s')
    results.sort(key=lambda r:r['cv_'+args.selection_metric])
    write_csv(OUT/'cv_summary.csv',results);write_csv(OUT/'cv_folds.csv',folds)
    eligible=[r for r in results if args.model_mode=='auto' or
              (r['kind']=='median_spline')==(args.model_mode=='median')]
    chosen=eligible[0]
    best_mean=next(r for r in results if r['kind']!='median_spline')
    best_median=next(r for r in results if r['kind']=='median_spline')
    log(f'\n[2/3] Refit selected model on all {len(train)} training trials: '
        f'{chosen["kind"]}, param={chosen["param"]:g}')
    model=fit(train,chosen['kind'],chosen['param'])
    mean_reference=model if model['kind']!='median_spline' else fit(train,best_mean['kind'],best_mean['param'])
    median_reference=model if model['kind']=='median_spline' else fit(train,best_median['kind'],best_median['param'])
    settings_path=args.source/train[0]['id']/'settings.json'
    if settings_path.is_file():
        recording_settings=json.loads(settings_path.read_text(encoding='utf-8-sig'))
        for saved_model in (model,mean_reference,median_reference):
            saved_model['recording_settings']=recording_settings
            saved_model['recording_settings_source']=str(settings_path.resolve())
    fitting_seconds=time.perf_counter()-fitting_started
    (OUT/'model.json').write_text(json.dumps(model,indent=2),encoding='utf-8')
    (OUT/'models'/'best_model.json').write_text(json.dumps(model,indent=2),encoding='utf-8')
    (OUT/'models'/'mean_reference.json').write_text(json.dumps(mean_reference,indent=2),encoding='utf-8')
    (OUT/'models'/'median_reference.json').write_text(json.dumps(median_reference,indent=2),encoding='utf-8')
    log(f'  {len(model["coefficients"])} fitted coefficients | fitting and validation: {fitting_seconds:.3f}s')
    if model.get('knot_strategy'):
        log(f'  Knot placement: {model["knot_strategy"]} (training data only)')
    log(f'\n[3/3] Evaluate {len(test)} held-out trials and save prediction plots')
    # Held-out evaluation starts only after method selection and final fitting.
    baseline=fit(train,'mean',0)
    reports=[];predictions=[]
    fig,axs=plt.subplots(len(test),1,figsize=(12,max(4,3.2*len(test))),squeeze=False)
    axs=axs[:,0]
    for ax,g in zip(axs,test):
        p=predict(model,g['x']);base=predict(baseline,g['x']);m=metrics(g['y'],p)
        b=metrics(g['y'],base)
        outside=(g['x']<model['x_min'])|(g['x']>model['x_max'])
        reports.append(dict(trial=g['id'],frames=len(p),**m,baseline_mae=b['mae'],baseline_rmse=b['rmse'],outside_training_range=int(outside.sum())))
        t=np.array([float(r['segment_elapsed_s']) for r in g['rows']])
        ax.plot(t,g['y'],label='Measured',color='#20242a',lw=2)
        ax.plot(t,p,label='Predicted',color='#d95035',lw=1.8)
        ax.axhline(base[0],label='Training mean baseline',color='#999999',ls=':',lw=1)
        ax.set_title(f'{g["id"]} | MAE {m["mae"]:.2f} px | RMSE {m["rmse"]:.2f} px')
        ax.set(xlabel='Time from segment start (s)',ylabel='Aperture (px)',ylim=(0,115));ax.grid(alpha=.2)
        single,single_ax=plt.subplots(figsize=(10,4))
        single_ax.plot(t,g['y'],label='Measured',color='#20242a',lw=2)
        single_ax.plot(t,p,label='Predicted',color='#d95035',lw=1.8)
        single_ax.set(xlabel='Time from segment start (s)',ylabel='Aperture (px)',
            title=f'{g["id"]} | MAE {m["mae"]:.2f} px | RMSE {m["rmse"]:.2f} px')
        single_ax.legend();single_ax.grid(alpha=.2);single.tight_layout()
        single.savefig(OUT/'test_predictions'/f'{g["id"]}.png',dpi=160);plt.close(single)
        log(f'  {g["id"]}: MAE={m["mae"]:.2f}px | RMSE={m["rmse"]:.2f}px | R2={m["r2"]:.3f}')
        for i,r in enumerate(g['rows']):
            predictions.append(dict(trial=g['id'],source_frame=r['frame'],time_s=t[i],distance_px=g['x'][i],
                actual_aperture_px=g['y'][i],predicted_aperture_px=p[i],error_px=p[i]-g['y'][i],outside_training_range=int(outside[i])))
    axs[0].legend(ncol=3)
    fig.suptitle(f'{len(test)} held-out trials | {model["kind"]}, parameter={model["param"]}',fontsize=16)
    fig.tight_layout(rect=(0,0,1,.975));fig.savefig(OUT/'test_predictions.png',dpi=150);plt.close(fig)
    write_csv(OUT/'test_metrics.csv',reports);write_csv(OUT/'test_predictions.csv',predictions)
    actual=np.concatenate([g['y'] for g in test]);pred=np.concatenate([predict(model,g['x']) for g in test])
    base=np.concatenate([predict(baseline,g['x']) for g in test])
    summary=dict(selected_by_training_cv=chosen,test_pooled_metrics=metrics(actual,pred),
        model_mode=args.model_mode,selection_metric=args.selection_metric,
        test_mean_trial_mae=float(np.mean([r['mae'] for r in reports])),
        test_mean_trial_rmse=float(np.mean([r['rmse'] for r in reports])),
        baseline_test_pooled_metrics=metrics(actual,base),baseline_value_px=baseline['coefficients'][0],
        outside_training_range_frames=sum(r['outside_training_range'] for r in reports),
        training_distance_range_px=[model['x_min'],model['x_max']])
    comparison=[]
    for label,reference,cv in [('Mean regression',mean_reference,best_mean),
                               ('Median regression',median_reference,best_median)]:
        rp=np.concatenate([predict(reference,g['x']) for g in test])
        comparison.append(dict(family=label,kind=reference['kind'],param=reference['param'],
            selected=int(reference['kind']==model['kind'] and reference['param']==model['param']),
            cv_mae=cv['cv_mae'],cv_rmse=cv['cv_rmse'],**metrics(actual,rp)))
    write_csv(OUT/'model_family_comparison.csv',comparison)
    compfig,compaxes=plt.subplots(1,3,figsize=(12,4.5))
    for ax,key,label in zip(compaxes,['mae','rmse','r2'],['MAE (px)','RMSE (px)','R\u00b2']):
        bars=ax.bar(['Mean','Median'],[r[key] for r in comparison],color=['#3485b8','#e59a32'])
        ax.bar_label(bars,fmt='%.3f' if key=='r2' else '%.2f',padding=3)
        ax.set_title(label);ax.margins(y=.2);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    compfig.suptitle('Same held-out trials | families tuned using training validation only')
    compfig.tight_layout(rect=(0,0,1,.92));compfig.savefig(OUT/'model_family_comparison.png',dpi=180);plt.close(compfig)
    plot_test_metrics(reports,summary['test_pooled_metrics'],OUT)
    (OUT/'evaluation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(11,7))
    for i,g in enumerate(train):ax.scatter(g['x'],g['y'],s=8,c='#b9c4cc',alpha=.35,label='Training frames' if i==0 else None)
    for i,g in enumerate(test):ax.scatter(g['x'],g['y'],s=13,c='#2683b5',alpha=.55,label='Test frames' if i==0 else None)
    grid=np.linspace(model['x_min'],model['x_max'],500)
    ax.plot(grid,predict(model,grid),color='#d95035',lw=2.5,label='Selected regression')
    ax.set(xlabel='Hand-object distance (px)',ylabel='Thumb-index aperture (px)',title=f'{args.source.parent.name} | {len(train)} training trials / {len(test)} test trials')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(OUT/'regression_fit.png',dpi=180);plt.close(fig)
    (OUT/'README.txt').write_text('Input: hand_object_distance (px); output: finger_width (px).\nSplit is frozen in split.json. Only training-trial leave-one-trial-out validation selects the model.\nSpline is a cubic B-spline with second-difference coefficient penalty. Polynomial uses a scaled input.\nmodel.json stores coefficients, scaling, and spline knots where applicable; predict() in the script loads this format.\nNo prediction clipping; no exported source measurements changed.\nTest trials share this session and participant; this does not establish cross-session/participant generalization.\n',encoding='utf-8')
    with (OUT/'README.txt').open('a',encoding='utf-8') as f:
        f.write(f'Model mode: {args.model_mode}; validation selection metric: {args.selection_metric}.\n'
                'Median spline estimates quantile 0.5 using weighted absolute error plus an L1 second-difference smoothness penalty.\n'
                'Mean and median references are each tuned on training folds, then compared on the identical test split.\n'
                'Model-family test comparison is descriptive and does not change the selected model.\n')
    summary['fitting_validation_seconds']=fitting_seconds
    summary['total_seconds']=time.perf_counter()-run_started
    (OUT/'evaluation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    write_csv(OUT/'overall_model_summary.csv',[dict(model=model['kind'],param=model['param'],
        **summary['test_pooled_metrics'])])
    score=summary['test_pooled_metrics']
    log('\n=== Test results (all held-out frames) ===')
    log(f'MAE: {score["mae"]:.2f}px | RMSE: {score["rmse"]:.2f}px | R2: {score["r2"]:.3f}')
    log(f'Baseline RMSE: {summary["baseline_test_pooled_metrics"]["rmse"]:.2f}px')
    log(f'Total time: {summary["total_seconds"]:.2f}s')
    log('\nSame test split comparison (not used to select the saved model):')
    for r in comparison:
        log(f'  {r["family"]}: MAE={r["mae"]:.2f}px | RMSE={r["rmse"]:.2f}px | R2={r["r2"]:.3f}')
    log(f'\nSaved results to: {OUT.resolve()}')
    for name in ['test_metrics_chart.png','test_metrics_chart.pdf','test_metrics_chart_data.csv',
                 'model_family_comparison.png','model_family_comparison.csv',
                 'test_predictions.png','test_predictions/*.png','regression_fit.png',
                 'overall_model_summary.csv','test_metrics.csv','test_predictions.csv',
                 'cv_summary.csv','data_split.csv','models/best_model.json','training_log.txt']:
        log('  '+name)
    (OUT/'RESULT.txt').write_text('\n'.join(log_lines)+'\n',encoding='utf-8')

if __name__=='__main__':main()
