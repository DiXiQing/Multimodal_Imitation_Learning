"""Plot per-grasp phase MAE from saved test predictions; no fitting required."""
import argparse
import csv
from pathlib import Path
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

MODELS = [('distance_window5','Window 5','predicted_window5_px')]
PHASES = [('rest','Initial','#338ac0'),('movement','Movement','#ff8510'),('hold','Hold','#2ca02c')]


def write_rows(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def export_phase_errors(output,predictions,prefix='test'):
    """Equal weight to each grasp's phase MAE; bars show sample SD, not SEM."""
    output=Path(output)
    trials=sorted({r['trial'] for r in predictions})
    per_trial=[]
    for trial in trials:
        for phase,_,_ in PHASES:
            subset=[r for r in predictions if r['trial']==trial and r['phase']==phase]
            if not subset:continue
            for model,_,key in MODELS:
                y=np.array([float(r['actual_aperture_px']) for r in subset])
                p=np.array([float(r[key]) for r in subset]);valid=np.isfinite(y)&np.isfinite(p)
                if not valid.any():continue
                per_trial.append(dict(model=model,trial=trial,phase=phase,frames=int(valid.sum()),
                                      mae_px=float(np.mean(abs(p[valid]-y[valid])))))
    if not per_trial:raise ValueError('No finite test predictions available for phase plot.')
    summary=[]
    for model,_,_ in MODELS:
        for phase,_,_ in PHASES:
            group=[r for r in per_trial if r['model']==model and r['phase']==phase]
            values=[r['mae_px'] for r in group]
            summary.append(dict(model=model,phase=phase,grasps=len(group),
                frames=sum(r['frames'] for r in group),
                mean_trial_mae_px=float(np.mean(values)) if values else None,
                sd_trial_mae_px=float(np.std(values,ddof=1)) if len(values)>1 else 0. if values else None))
    write_rows(output/f'{prefix}_trial_phase_metrics.csv',per_trial)
    write_rows(output/f'{prefix}_phase_summary.csv',summary)
    fig=Figure(figsize=(8,5.5));FigureCanvasAgg(fig);ax=fig.subplots()
    for j,(phase,label,color) in enumerate(PHASES):
        row=next(r for r in summary if r['phase']==phase)
        mean=row['mean_trial_mae_px'] if row['mean_trial_mae_px'] is not None else np.nan
        sd=row['sd_trial_mae_px'] if row['sd_trial_mae_px'] is not None else 0
        ax.bar(j,mean,width=.55,color=color,yerr=sd,capsize=5)
    ax.set_xticks(np.arange(3),[label for _,label,_ in PHASES])
    ax.set(title='Errors by phase | Window 5',ylabel='Mean per-grasp MAE (px)')
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    counts=sorted({r['grasps'] for r in summary})
    note=(f'Error bars: sample SD across {counts[0]} {prefix} grasps' if len(counts)==1 else
          f'Error bars: sample SD across available {prefix} grasps; counts in {prefix}_phase_summary.csv')
    fig.text(.5,.015,note,ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig(output/f'{prefix}_phase_errors.png',dpi=200)
    fig.savefig(output/f'{prefix}_phase_errors.pdf')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',required=True,type=Path)
    args=parser.parse_args()
    with (args.result/'test_predictions.csv').open(encoding='utf-8-sig') as f:
        predictions=list(csv.DictReader(f))
    summary=export_phase_errors(args.result,predictions)
    for row in summary:print(row)
    print('Saved:',args.result/'test_phase_errors.png')


if __name__=='__main__':main()
