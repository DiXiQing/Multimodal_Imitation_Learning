"""Raw distance-aperture overlays and phase diagnostics for top-view trials."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from trial_paths import grasp_folders


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,type=Path)
    args=parser.parse_args();source=args.source;out=source/'relationship_plots';out.mkdir(exist_ok=True)
    settings=json.loads((source.parent/'settings.json').read_text(encoding='utf-8-sig'))
    object_label=f'{settings.get("object_color","black").capitalize()} object {settings["size_cm"]:g} cm'
    with (source/'crop_summary.csv').open(encoding='utf-8-sig') as f:bounds={int(r['segment']):r for r in csv.DictReader(f)}
    trials=[];stats=[]
    for folder in grasp_folders(source):
        with (folder/'camera.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
        def values(col):return np.array([float(r[col]) if r[col] else np.nan for r in rows])
        x=values('hand_object_distance');y=values('finger_width');frames=values('frame');valid=values('valid')==1
        x[~valid]=np.nan;y[~valid]=np.nan
        n=int(json.loads((folder/'segment.json').read_text(encoding='utf-8-sig'))['segment'])
        b=bounds[n];onset=int(b['onset']);stable=int(b['stable'])
        approach=(frames>=onset)&(frames<=stable);hold=frames>=stable
        trials.append(dict(n=n,x=x,y=y,frames=frames,approach=approach,hold=hold))
        stats.append(dict(trial=n,frames=len(rows),invalid_frames=int((~valid).sum()),start_distance_px=float(x[0]),
            peak_aperture_px=float(np.nanmax(y)),distance_at_peak_px=float(x[np.nanargmax(y)]),
            hold_median_aperture_px=float(np.nanmedian(y[hold])),hold_median_distance_px=float(np.nanmedian(x[hold])),
            object_center_x_range_px=float(np.ptp(values('object_x'))),object_center_y_range_px=float(np.ptp(values('object_y')))))
    if not trials:parser.error('No grasp trial directories found.')
    colors=plt.cm.turbo(np.linspace(.03,.97,len(trials)))
    fig,ax=plt.subplots(figsize=(12.5,7.5))
    for g,color in zip(trials,colors):ax.plot(g['x'],g['y'],'o-',color=color,lw=1.3,markersize=2.5,alpha=.85,label=f'Trial {g["n"]:02d}')
    ax.set(xlabel='Hand-object distance (px)',ylabel='Thumb-index aperture (px)',title=f'{object_label} | {len(trials)} trials')
    ax.grid(alpha=.2);ax.legend(ncol=3 if len(trials)>32 else 2,loc='upper left',bbox_to_anchor=(1.01,1),fontsize=9,frameon=False)
    fig.text(.08,.025,'Raw valid measurements, connected in time order. Includes rest, approach and hold. No smoothing or alignment.',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(out/'distance_aperture_overlay.png',dpi=180);fig.savefig(out/'distance_aperture_overlay.pdf');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(15,6))
    for g,color in zip(trials,colors):
        a=g['approach'];h=g['hold'];axs[0].plot(g['x'][a],g['y'][a],'o-',color=color,ms=3,lw=1)
        axs[1].plot(g['x'][h],g['y'][h],'o-',color=color,ms=3,lw=1)
    axs[0].set_title('Movement onset to first stable grasp')
    axs[1].set_title('Grasp hold (from segmentation boundary)')
    for ax in axs:ax.set(xlabel='Hand-object distance (px)',ylabel='Aperture (px)');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'phase_diagnostics.png',dpi=170);plt.close(fig)
    nrows=(len(trials)+3)//4
    fig,axs=plt.subplots(nrows,4,figsize=(15,3.2*nrows),sharex=True,sharey=True,squeeze=False)
    for ax,g,color in zip(axs.flat,trials,colors):
        ax.plot(g['x'],g['y'],'o-',color=color,ms=2.5,lw=1)
        ax.set_title(f'Trial {g["n"]:02d}');ax.grid(alpha=.2)
    for ax in list(axs.flat)[len(trials):]:ax.set_visible(False)
    fig.supxlabel('Hand-object distance (px)');fig.supylabel('Aperture (px)');fig.tight_layout(rect=(.02,.02,1,1));fig.savefig(out/'individual_trials.png',dpi=150);plt.close(fig)
    with (out/'trial_diagnostics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(stats[0]));w.writeheader();w.writerows(stats)
    summary={key:[min(r[key] for r in stats),max(r[key] for r in stats)] for key in list(stats[0])[3:]}
    print(json.dumps(summary,indent=2));print('Output:',out)

if __name__=='__main__':main()
