"""Summarize existing live predictions, without modifying or rerunning them."""
from pathlib import Path
from datetime import datetime
import csv,json,hashlib,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]/'Data/D4/Result/fixed_function_mlp_lstm_window5_20261026_205550_700402/models/RealTime'
MODELS=[('function','predicted_function_px','Function'),('mlp','predicted_mlp_px','MLP (5 frames)'),('lstm','predicted_lstm_px','LSTM (full history)')]
GROUPS=[('fast','Fast'),('normal','Medium'),('slow','Slow')]
COLORS=['#3485b8','#e59a32','#519e58']
GROUP_COLORS=['#d94c43','#3485b8','#555555']

def write(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def scores(y,p):
    e=p-y;den=np.sum((y-y.mean())**2)
    return dict(mae=float(abs(e).mean()),rmse=float(np.sqrt((e*e).mean())),r2=float(1-(e*e).sum()/den) if den>1e-12 else None,bias=float(e.mean()))

def save(fig,out,name):
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'{name}.{ext}',dpi=200)
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT)
    parser.add_argument('--output-root',type=Path)
    args=parser.parse_args()
    source=args.source.resolve()
    out=(args.output_root or source)/f'summary_{datetime.now():%Y%m%d_%H%M%S_%f}';out.mkdir(parents=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    trials=[];metrics=[];coverage=[];hashes={}
    for group,label in GROUPS:
        folders=sorted(p.parent for p in (source/group).glob('*/predictions.csv'))
        for i,folder in enumerate(folders,1):
            path=folder/'predictions.csv';hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            with path.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
            def column(key):return np.array([float(r[key]) if r.get(key) else np.nan for r in rows])
            y=column('finger_width');t=column('t_camera');t=t-t[0]
            pred=np.column_stack([column(key) for _,key,_ in MODELS])
            valid=np.array([r.get('valid','').lower() in ('true','1') and r.get('common_valid','').lower() in ('true','1') for r in rows])
            valid &= np.isfinite(y)&np.isfinite(pred).all(axis=1)
            if not valid.any():raise ValueError(f'No common valid frames: {folder}')
            trial=dict(group=group,group_label=label,label=f'{label} {i}',folder=folder.name,y=y,t=t,pred=pred,mask=valid,rows=rows)
            trials.append(trial)
            coverage.append(dict(condition=group,label=trial['label'],trial=folder.name,retained_rows=len(rows),common_frames=int(valid.sum()),duration_s=float(t[-1]),
                                 csv_sha256=hashes[str(path)],manual_crop_info_present=(folder/'manual_crop_info.json').exists()))
            for j,(name,key,model_label) in enumerate(MODELS):
                metrics.append(dict(condition=group,label=trial['label'],trial=folder.name,model=name,frames=int(valid.sum()),**scores(y[valid],pred[valid,j])))
    if len(trials)!=9:raise ValueError(f'Expected 9 selected trials; found {len(trials)}')
    overall=[];by_speed=[]
    for group in ['Overall']+[g for g,_ in GROUPS]:
        part=trials if group=='Overall' else [tr for tr in trials if tr['group']==group]
        y=np.concatenate([tr['y'][tr['mask']] for tr in part])
        for j,(name,key,label) in enumerate(MODELS):
            p=np.concatenate([tr['pred'][tr['mask'],j] for tr in part])
            row=dict(condition=group,model=name,trials=len(part),frames=len(y),**scores(y,p))
            (overall if group=='Overall' else by_speed).append(row)
    write(out/'overall_metrics.csv',overall);write(out/'metrics_by_speed.csv',by_speed)
    write(out/'trial_metrics.csv',metrics);write(out/'trial_coverage.csv',coverage)
    fig,axes=plt.subplots(1,3,figsize=(13,4.5))
    for ax,key,title in zip(axes,['mae','rmse','r2'],['MAE (px)','RMSE (px)','R²']):
        bars=ax.bar([label for _,_,label in MODELS],[r[key] if r[key] is not None else np.nan for r in overall],color=COLORS)
        ax.bar_label(bars,fmt='%.3f' if key=='r2' else '%.2f',padding=4);ax.set_title(title);ax.tick_params(axis='x',labelsize=9);ax.margins(y=.25);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    save(fig,out,'01_overall_model_comparison')
    fig,axes=plt.subplots(3,3,figsize=(13,10.5))
    for row_axes,(group,label) in zip(axes,GROUPS):
        rows=[r for r in by_speed if r['condition']==group]
        for ax,key,title in zip(row_axes,['mae','rmse','r2'],['MAE (px)','RMSE (px)','R²']):
            bars=ax.bar([label for _,_,label in MODELS],[r[key] if r[key] is not None else np.nan for r in rows],color=COLORS)
            ax.bar_label(bars,fmt='%.3f' if key=='r2' else '%.2f',padding=3);ax.set_title(f'{label}: {title}');ax.tick_params(axis='x',labelsize=8);ax.margins(y=.25);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    save(fig,out,'02_metrics_by_speed')
    for number,(name,_,model_label) in enumerate(MODELS,3):
        rows=[r for r in metrics if r['model']==name]
        fig,axes=plt.subplots(1,3,figsize=(15,4.5))
        colors=[GROUP_COLORS[[g for g,_ in GROUPS].index(r['condition'])] for r in rows]
        for ax,key,title in zip(axes,['mae','rmse','r2'],['MAE (px)','RMSE (px)','R²']):
            bars=ax.bar([r['label'] for r in rows],[r[key] if r[key] is not None else np.nan for r in rows],color=colors)
            ax.bar_label(bars,fmt='%.3f' if key=='r2' else '%.2f',padding=3,fontsize=8)
            ax.set_title(title);ax.tick_params(axis='x',rotation=50,labelsize=9);ax.margins(y=.22);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        save(fig,out,f'{number:02d}_trial_metrics_{name}')
    fig,axes=plt.subplots(3,3,figsize=(15,10),sharey=True)
    handles=None
    for ax,tr in zip(axes.flat,trials):
        measured=tr['y'].copy();measured[np.array([r.get('valid','').lower() not in ('true','1') for r in tr['rows']])]=np.nan
        ax.plot(tr['t'],measured,color='#222222',lw=1.8,label='Measured')
        for j,(_,_,label) in enumerate(MODELS):ax.plot(tr['t'],tr['pred'][:,j],color=COLORS[j],lw=1.3,label=label)
        ax.set_title(f'{tr["label"]} | {tr["folder"].split("_")[2]}',fontsize=10);ax.grid(alpha=.18)
        handles=ax.get_legend_handles_labels()
    fig.legend(*handles,loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.5,1.005))
    fig.supxlabel('Time from retained segment start (s)');fig.supylabel('Aperture (px)');fig.tight_layout(rect=(.02,.02,1,.96))
    for ext in ('png','pdf'):fig.savefig(out/f'06_prediction_trajectories.{ext}',dpi=200)
    plt.close(fig)
    mean_sd=[]
    fig,axes=plt.subplots(1,3,figsize=(13,4.5))
    for ax,key,title in zip(axes,['mae','rmse','r2'],['Trial MAE (px)','Trial RMSE (px)','Trial R²']):
        means=[];sds=[]
        for name,_,label in MODELS:
            values=[r[key] for r in metrics if r['model']==name and r[key] is not None]
            mean=float(np.mean(values));sd=float(np.std(values,ddof=1))
            mean_sd.append(dict(model=name,metric=key,trials=len(values),mean=mean,sample_sd=sd))
            means.append(mean);sds.append(sd)
        bars=ax.bar([label for _,_,label in MODELS],means,yerr=sds,capsize=5,color=COLORS)
        for bar,mean,sd in zip(bars,means,sds):ax.text(bar.get_x()+bar.get_width()/2,mean+sd,f'{mean:.3f}' if key=='r2' else f'{mean:.2f}',ha='center',va='bottom',fontsize=10)
        ax.set_title(title);ax.tick_params(axis='x',labelsize=9);ax.margins(y=.25);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    save(fig,out,'07_trial_mean_sd');write(out/'trial_mean_sd.csv',mean_sd)
    method=dict(trials=len(trials),source=str(source),grouping='Folder labels fast/normal/slow (inferred approach duration grouping from original selection; original target metadata was 1 s for all trials).',
        evaluation='Common_valid AND valid AND finite measured/predicted values for all three models. Same rows for all model scores. Overall and speed scores pool frames; mean±SD figure averages per-trial scores with sample SD ddof=1.',
        predictions='Existing saved predictions only, not rerun. Any pre-crop recurrent history is retained.',
        scope='Nine previously selected best examples; not an unbiased summary of all live experiments.',units='px; no mm calibration supplied.',source_hashes=hashes)
    (out/'method.json').write_text(json.dumps(method,indent=2),encoding='utf-8')
    for path,h in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
    print('OUTPUT',out)
    for r in overall:print(r)

if __name__=='__main__':main()
