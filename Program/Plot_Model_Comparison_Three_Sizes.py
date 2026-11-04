"""Three-size model comparison from saved out-of-fold trial metrics; no training."""
from pathlib import Path
import csv,json,os
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).parent/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'Data').is_dir())
SOURCES={20:ROOT/'Data/1101_D/Result/20mm_20261103_185038',30:ROOT/'Data/1101_D/Result/30mm_Combined_BSpline_MLP5_LSTMFull_20261103_195651',40:ROOT/'Data/1101_D/Result/40mm_20261103_172526'}
LABELS=['Function','MLP 5f','LSTM Full'];COLORS=['#3485b8','#e59a32','#519e58']
OUT=ROOT/'Data/1101_D/Result/three_sizes_comparison_20261103'

def main():
    summary=[]
    for size,folder in SOURCES.items():
        with (folder/'oof_per_trial_metrics.csv').open(encoding='utf-8-sig',newline='') as f:rows=[r for r in csv.DictReader(f) if r['split']=='Test']
        reference=None
        for model in LABELS:
            selected=[r for r in rows if r['model']==model];names=[r['trial'] for r in selected]
            assert len(names)==len(set(names))==200
            if reference is None:reference=set(names)
            else:assert set(names)==reference
            for metric in ['mae_mm','rmse_mm','r2']:
                v=np.array([float(r[metric]) for r in selected]);assert np.isfinite(v).all()
                summary.append(dict(size_mm=size,model=model,metric=metric,n_trials=len(v),mean=float(v.mean()),sd=float(v.std(ddof=1))))
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'mean_sd.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    def draw(ax,key,label):
        x=np.arange(3)*1.4;width=.28;tops=[]
        for j,(model,color) in enumerate(zip(LABELS,COLORS)):
            rr=[next(r for r in summary if r['size_mm']==size and r['model']==model and r['metric']==key) for size in SOURCES]
            means=np.array([r['mean'] for r in rr]);sd=np.array([r['sd'] for r in rr]);pos=x+(j-1)*width
            ax.bar(pos,means,width=width,color=color,yerr=sd,capsize=3,label=model,error_kw={'elinewidth':1.1,'capthick':1.1})
            for k,(px,y,e) in enumerate(zip(pos,means,sd)):
                ax.annotate(f'{y:.3f}' if key=='r2' else f'{y:.2f}',(px,y+e),xytext=(0,5+10*j if key=='r2' else 5),textcoords='offset points',ha='center',fontsize=8)
            tops.extend(means+sd)
        ax.set_xticks(x,['20 mm','30 mm','40 mm']);ax.set_ylabel('');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False);ax.set_ylim(0,max(tops)*1.15)
    for key,label in [('mae_mm','MAE (mm)'),('rmse_mm','RMSE (mm)'),('r2','R²')]:
        fig,ax=plt.subplots(figsize=(5,4.2));draw(ax,key,label);fig.tight_layout()
        for ext in ('png','svg','pdf'):fig.savefig(OUT/f'{key}.{ext}',dpi=240,bbox_inches='tight')
        plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(14,4.1))
    for ax,(key,label) in zip(axs,[('mae_mm','MAE (mm)'),('rmse_mm','RMSE (mm)'),('r2','R²')]):draw(ax,key,label)
    fig.tight_layout()
    for ext in ('png','svg','pdf'):fig.savefig(OUT/f'three_sizes_model_comparison.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)
    (OUT/'sources.json').write_text(json.dumps(dict(sources={str(k):str(v) for k,v in SOURCES.items()},definition='Equal-trial mean and sample SD (ddof=1) across 200 out-of-fold Test trials per size; common valid target frames. Bars show means; error bars SD.',function_note='30mm Function uses fixed-knot cubic B-spline regression from specified combined run; 20mm and 40mm use their original smoothing-spline Function. No claim of identical function specification across sizes.'),indent=2),encoding='utf-8')
    print('Saved:',OUT)

if __name__=='__main__':main()
