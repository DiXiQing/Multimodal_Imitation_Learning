"""Summarize saved D1 validation predictions; never fit models."""
import csv
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
SOURCES={
    'Red 20 mm':ROOT/'New/Data/D1/2/Result/distance_mlp_window5_20261018_230453_976437',
    'Blue 30 mm':ROOT/'New/Data/D1/3/Result/distance_mlp_window5_20261018_221011_316327',
    'Black 40 mm':ROOT/'New/Data/D1/4/Result/distance_mlp_window5_20261018_225943_941141',
}
MODELS=[('distance_current','Current distance','predicted_current_px','#3485b8'),
        ('distance_window5','5-frame window','predicted_window5_px','#e59a32')]
PHASES=[('rest','Initial'),('movement','Grasping'),('hold','Hold')]


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def metrics(rows,key):
    y=np.array([float(r['actual_aperture_px']) for r in rows]);p=np.array([float(r[key]) for r in rows])
    if not len(y) or not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError('Missing/nonfinite saved predictions; inspect source before pooling.')
    e=p-y;den=np.sum((y-y.mean())**2)
    return dict(frames=len(y),mae=float(np.mean(abs(e))),rmse=float(np.sqrt(np.mean(e**2))),
                r2=float(1-np.sum(e**2)/den) if den>1e-12 else None)


def save_figure(fig,out,name):
    fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=200)
    fig.savefig(out/(name+'.pdf'));plt.close(fig)


def phase_bars(ax,summary,target):
    for j,(model,label,key,color) in enumerate(MODELS):
        group=[next(r for r in summary if r['target']==target and r['model']==model and r['phase']==phase)
               for phase,_ in PHASES]
        ax.bar(np.arange(3)+(j-.5)*.32,[r['mean_trial_mae_px'] for r in group],width=.30,color=color,
               yerr=[r['sd_trial_mae_px'] for r in group],capsize=4,label=label)
    ax.set_xticks(np.arange(3),[label for _,label in PHASES]);ax.set_ylabel('Mean per-grasp MAE (px)')
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)


def main():
    out=ROOT/'New/Result'/f'D1_validation_summary_{datetime.now():%Y%m%d_%H%M%S}'
    out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for target,path in SOURCES.items():
        with (path/'validation_predictions.csv').open(encoding='utf-8-sig') as f:
            source=list(csv.DictReader(f))
        for r in source:r['target']=target
        rows.extend(source)
    groups=[*SOURCES,'Overall'];overall=[];trial_phase=[];phase_summary=[]
    for target in groups:
        subset=[r for r in rows if target=='Overall' or r['target']==target]
        ids=sorted({(r['target'],r['trial']) for r in subset})
        for model,label,key,color in MODELS:
            overall.append(dict(target=target,model=model,grasps=len(ids),**metrics(subset,key)))
            if target!='Overall':
                for group,trial in ids:
                    for phase,_ in PHASES:
                        part=[r for r in subset if r['target']==group and r['trial']==trial and r['phase']==phase]
                        if part:trial_phase.append(dict(target=group,trial=trial,model=model,phase=phase,**metrics(part,key)))
    for target in groups:
        for model,label,key,color in MODELS:
            for phase,_ in PHASES:
                part=[r for r in trial_phase if (target=='Overall' or r['target']==target) and r['model']==model and r['phase']==phase]
                values=[r['mae'] for r in part]
                if not values:raise ValueError(f'No phase data: {target}/{model}/{phase}')
                phase_summary.append(dict(target=target,model=model,phase=phase,grasps=len(values),
                    frames=sum(r['frames'] for r in part),mean_trial_mae_px=float(np.mean(values)),
                    sd_trial_mae_px=float(np.std(values,ddof=1)) if len(values)>1 else 0.))
    write_csv(out/'overall_and_target_metrics.csv',overall)
    write_csv(out/'per_grasp_phase_metrics.csv',trial_phase)
    write_csv(out/'phase_summary.csv',phase_summary)
    # All three targets combined; recompute metrics from concatenated frames.
    fig,axes=plt.subplots(1,3,figsize=(12,4.5))
    for ax,key,title in zip(axes,['mae','rmse','r2'],['MAE (px)','RMSE (px)','R²']):
        values=[next(r for r in overall if r['target']=='Overall' and r['model']==m)[key] for m,_,_,_ in MODELS]
        bars=ax.bar(['Current','Window 5'],values,color=[c for _,_,_,c in MODELS]);ax.bar_label(bars,fmt='%.3f' if key=='r2' else '%.2f',padding=3)
        ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.margins(y=.2)
    save_figure(fig,out,'01_overall_model_comparison')
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    for ax,key,title in zip(axes,['mae','rmse','r2'],['MAE (px)','RMSE (px)','R²']):
        for j,(model,label,_,color) in enumerate(MODELS):
            values=[next(r for r in overall if r['target']==g and r['model']==model)[key] for g in groups]
            ax.bar(np.arange(4)+(j-.5)*.32,values,width=.30,color=color,label=label)
        ax.set_xticks(np.arange(4),['Red\n20 mm','Blue\n30 mm','Black\n40 mm','Overall'])
        ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(fontsize=9);save_figure(fig,out,'02_target_and_overall_metrics')
    fig,ax=plt.subplots(figsize=(8,5.5));phase_bars(ax,phase_summary,'Overall')
    ax.set_title('Errors by phase');ax.legend();save_figure(fig,out,'03_overall_phase_comparison')
    fig,axes=plt.subplots(1,3,figsize=(15,5),sharey=True)
    for ax,target in zip(axes,SOURCES):
        phase_bars(ax,phase_summary,target);ax.set_title(target)
    axes[0].legend(fontsize=9);save_figure(fig,out,'04_phase_comparison_by_target')
    (out/'sources_and_method.json').write_text(json.dumps(dict(sources={k:str(v) for k,v in SOURCES.items()},
        total_grasps=len({(r['target'],r['trial']) for r in rows}),total_frames=len(rows),
        overall='MAE/RMSE/R2 recomputed on concatenated eligible validation frames; not mean of target metrics. R2 uses the global measured mean.',
        phase='Mean of individual-grasp phase MAEs; error bars are sample SD (ddof=1) across 27 grasps overall or 9 grasps per target. No footer on plots.',
        note='Saved validation predictions only. No retraining, smoothing, exclusions, or changes to source results. Pixel scales refer to respective camera recordings.'),indent=2),encoding='utf-8')
    for r in overall:
        if r['target']=='Overall':print(r)
    print('Saved:',out)


if __name__=='__main__':main()
