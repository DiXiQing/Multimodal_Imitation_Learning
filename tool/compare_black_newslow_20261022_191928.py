from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path.cwd();source=root/'Data/TopView/TRIAL_Color_black_Size_4cm_20261022_191928_869177';out=root/'New/Result/black_fast_newslow_20261022_191928';out.mkdir(exist_ok=False)
groups=[('Fast (1 s cue)',root/'Data/TopView/TRIAL_Color_black_Size_4cm_20261022_161759_111243/grasp_segments','#3485b8'),('New slower (unpaced)',source/'grasp_segments','#e59a32')];allmetrics=[];datasets={}
for label,folder,color in groups:
 datasets[label]=[]
 for p in sorted(folder.glob('*/camera.csv')):
  rows=list(csv.DictReader(p.open(encoding='utf-8-sig')));meta=json.loads((p.parent/'segment.json').read_text());t=np.array([float(r['t_camera']) for r in rows]);t-=t[0]
  b=np.array([float(r['finger_width']) if r['finger_width'] else np.nan for r in rows]);d=np.array([float(r['hand_object_distance']) if r['hand_object_distance'] else np.nan for r in rows]);frames=np.array([int(r['frame']) for r in rows]);base=int(meta.get('source_start_frame',frames[0]))-int(meta['start']);onset_id=base+int(meta['onset']);stable_id=base+int(meta['stable'])
  oi=int(np.argmin(abs(frames-onset_id)));si=int(np.argmin(abs(frames-stable_id)));peak=int(np.nanargmax(b))
  metrics=dict(group=label,trial=p.parent.name,frames=len(rows),segment_duration_s=float(t[-1]),onset_to_stable_s=float(t[si]-t[oi]),peak_aperture_px=float(b[peak]),hold_aperture_px=float(np.nanmedian(b[si:])),initial_distance_px=float(np.nanmedian(d[:min(8,len(d))])),remaining_distance_at_peak_percent=float((d[peak]-d[-1])/(d[0]-d[-1])*100),invalid_frames=sum(r['valid']!='1' for r in rows));allmetrics.append(metrics);datasets[label].append((t,b,d))
summary=[]
for label,folder,color in groups:
 part=[r for r in allmetrics if r['group']==label]
 for key in ['frames','segment_duration_s','onset_to_stable_s','peak_aperture_px','hold_aperture_px','initial_distance_px','remaining_distance_at_peak_percent']:
  v=np.array([r[key] for r in part]);summary.append(dict(group=label,metric=key,mean=float(np.mean(v)),sample_sd=float(np.std(v,ddof=1)),trials=len(part)))
for name,data in [('speed_comparison_per_trial',allmetrics),('speed_comparison_summary',summary)]:
 with (out/(name+'.csv')).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
fig,axes=plt.subplots(1,3,figsize=(15,4.5))
for label,folder,color in groups:
 for i,(t,b,d) in enumerate(datasets[label]):
  axes[0].plot(t,b,color=color,alpha=.35,lw=1,label=label if i==0 else None);axes[1].plot(t,d,color=color,alpha=.35,lw=1);axes[2].plot(d,b,color=color,alpha=.35,lw=1)
axes[0].set(xlabel='Segment time (s)',ylabel='Fingertip aperture (px)');axes[1].set(xlabel='Segment time (s)',ylabel='Hand-object distance (px)');axes[2].set(xlabel='Hand-object distance (px)',ylabel='Fingertip aperture (px)');axes[0].legend()
for ax in axes:ax.grid(alpha=.15)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(out/f'fast_vs_slower_comparison.{ext}',dpi=200)
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4))
for i,(t,b,d) in enumerate(datasets['New slower (unpaced)']):
 color=plt.cm.turbo(i/max(1,len(datasets['New slower (unpaced)'])-1));axes[0].plot(t,b,color=color,lw=1,alpha=.7);axes[1].plot(t,d,color=color,lw=1,alpha=.7)
for ax,ylabel in zip(axes,['Fingertip aperture (px)','Hand-object distance (px)']):ax.set(xlabel='Segment time (s)',ylabel=ylabel);ax.grid(alpha=.15)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(out/f'slow_trials_overlay.{ext}',dpi=200)
plt.close(fig)
(out/'comparison_method.json').write_text(json.dumps(dict(sources={label:str(folder) for label,folder,_ in groups},note='Descriptive comparison only, no training. Actual CSV timestamps, no time normalization or truncation. Onset/stable use saved event annotations; not metronome-defined movement duration. Both settings label black4cm; Fast metronome target 1 second; new slow recording has metronome disabled. Different recording sessions may also differ in behavior. Missing original measurements preserved.',invalid_new=sum(r['invalid_frames'] for r in allmetrics if r['group']=='New slower (unpaced)')),indent=2))
print(json.dumps(summary,indent=2))

# Also place the two-panel overlay alongside each recording's crops.
for label,folder,color in groups:
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for i,(t,b,d) in enumerate(datasets[label]):
        c=plt.cm.turbo(i/max(1,len(datasets[label])-1))
        axes[0].plot(t,b,color=c,lw=1,alpha=.7)
        axes[1].plot(t,d,color=c,lw=1,alpha=.7)
    for ax,ylabel in zip(axes,['Fingertip aperture (px)','Hand-object distance (px)']):
        ax.set(xlabel='Segment time (s)',ylabel=ylabel);ax.grid(alpha=.15)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(folder/f'grasp_overlay.{ext}',dpi=200)
    plt.close(fig)
