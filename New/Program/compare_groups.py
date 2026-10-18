"""Compare two collections of grasp trials using one color per collection."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def load_group(root):
    trials=[]
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not (folder/'camera.csv').is_file():continue
        with (folder/'camera.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
        rows=[r for r in rows if r.get('valid')=='1' and r.get('hand_object_distance') and r.get('finger_width')]
        if not rows:continue
        x=np.array([float(r['hand_object_distance']) for r in rows]);y=np.array([float(r['finger_width']) for r in rows])
        frames=np.array([int(r['frame']) for r in rows])
        meta=json.loads((folder/'segment.json').read_text(encoding='utf-8-sig'))
        settings=json.loads((folder/'settings.json').read_text(encoding='utf-8-sig'))
        hold=frames>=meta['stable']
        trials.append(dict(name=folder.name,x=x,y=y,settings=settings,peak=float(y.max()),
            peak_distance=float(x[y.argmax()]),start_distance=float(x[0]),hold=float(np.median(y[hold]))))
    return trials


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--group1',type=Path,required=True);p.add_argument('--group2',type=Path,required=True);p.add_argument('--group3',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    paths=[args.group1,args.group2]+([args.group3] if args.group3 else [])
    groups=[load_group(path) for path in paths];assert all(groups)
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    colors=['#2378b5','#e58223','#27965b'][:len(groups)];names=[f'Folder {i+1}' for i in range(len(groups))]
    fig,axs=plt.subplots(1,2,figsize=(15,6.5))
    binrows=[];stats=[]
    for name,trials,color in zip(names,groups,colors):
        for g in trials:
            axs[0].plot(g['x'],g['y'],color=color,alpha=.23,lw=1)
            stats.append(dict(group=name,trial=g['name'],valid_frames=len(g['x']),peak_aperture_px=g['peak'],distance_at_peak_px=g['peak_distance'],hold_median_aperture_px=g['hold'],start_distance_px=g['start_distance']))
        centers=[];medians=[];lower=[];upper=[]
        for left in range(0,260,10):
            per_trial=[]
            for g in trials:
                take=(g['x']>=left)&(g['x']<left+10)
                if take.any():per_trial.append(float(np.median(g['y'][take])))
            if not per_trial:continue
            lo,mid,hi=np.percentile(per_trial,[25,50,75])
            binrows.append(dict(group=name,distance_start_px=left,distance_end_px=left+10,trials=len(per_trial),median_px=mid,q25_px=lo,q75_px=hi))
            if len(per_trial)>=3:centers.append(left+5);medians.append(mid);lower.append(lo);upper.append(hi)
        axs[1].plot(centers,medians,color=color,lw=2.2,label=f'{name}: median')
        axs[1].fill_between(centers,lower,upper,color=color,alpha=.18)
    for ax in axs:ax.set(xlabel='Hand-object distance (px)',ylabel='Thumb-index aperture (px)',xlim=(-5,265),ylim=(0,112));ax.grid(alpha=.2)
    axs[0].set_title('All raw grasp curves | one color per experiment')
    axs[1].set_title('10 px bins | median and interquartile range across trials')
    axs[0].legend(handles=[Line2D([0],[0],color=c,lw=2,label=f'{n}: {len(g)} trials') for n,g,c in zip(names,groups,colors)],loc='lower center')
    axs[1].legend(loc='lower center')
    object_types={(g['settings'].get('object_color','black'),g['settings']['size_cm']) for group in groups for g in group}
    label=' / '.join(f'{color.capitalize()} {size:g} cm' for color,size in sorted(object_types))
    fig.suptitle(label+' | '+' vs '.join(names),fontsize=16)
    fig.text(.5,.02,'Raw curves are unchanged. Right panel uses one median per trial per bin, giving trials equal weight; bands are not confidence intervals.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.94));fig.savefig(out/'group_comparison.png',dpi=185);fig.savefig(out/'group_comparison.pdf');plt.close(fig)
    for filename,rows in [('trial_summary.csv',stats),('distance_bin_summary.csv',binrows)]:
        with (out/filename).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for name,trials in zip(names,groups):
        r=dict(group=name,trials=len(trials),valid_frames=sum(len(g['x']) for g in trials))
        for key in ['peak','peak_distance','hold','start_distance']:
            a=np.array([g[key] for g in trials]);r[key+'_mean']=float(a.mean());r[key+'_sd']=float(a.std(ddof=1))
        r['recording_configurations']=[dict(t) for t in {json.dumps(g['settings'],sort_keys=True):g['settings'] for g in trials}.values()]
        summary.append(r)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps([{k:v for k,v in r.items() if k!='recording_configurations'} for r in summary],indent=2))
    print('Dimensions:',[sorted({(g['settings']['width'],g['settings']['height']) for g in group}) for group in groups])
    print('Saved:',out.resolve())

if __name__=='__main__':main()
