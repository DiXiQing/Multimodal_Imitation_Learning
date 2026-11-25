from pathlib import Path
import csv,json
import numpy as np
import cv2
from scipy.signal import medfilt,find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(r'C:\MineApp\Code\Multimodal_Imitation_Learning');base=ROOT/'Data/Marker_Realtime';out=ROOT/'Report/Marker_Realtime/Review_20261124_1820';out.mkdir(parents=True,exist_ok=True)
trials=[p for p in sorted(base.glob('marker_20261124_182*')) if 'marker_20261124_182020'<=p.name<='marker_20261124_182351_999999']
print('TRIAL_COUNT',len(trials),flush=True)
records=[];loaded=[]
for number,p in enumerate(trials,1):
 with (p/'predictions.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
 m=json.loads((p/'metadata.json').read_text());v=lambda k:np.array([float(r[k]) for r in rows]);t=v('time_s');d=v('distance_mm');valid=np.isfinite(d)&(v('marker_valid')==1);dt=np.diff(t);fps=(len(t)-1)/(t[-1]-t[0]);finite=np.isfinite(v('lstm_mm'))
 smooth=medfilt(np.interp(t,t[valid],d[valid]),5);initial=float(np.median(smooth[:min(8,len(t))]));final=float(np.median(smooth[t>=t[-1]-.5]));minimum=float(smooth.min());amplitude=initial-minimum;threshold=max(5,amplitude*.03)
 onset_candidates=np.flatnonzero(smooth<initial-threshold);onset=int(onset_candidates[0]) if len(onset_candidates) else None
 # Robust near-object arrival: first crossing within 5 mm of terminal distance, when terminal distance is near the minimum.
 finish_candidates=np.flatnonzero((smooth<=final+5)&(np.arange(len(t))>(onset or 0)))
 finish=int(finish_candidates[0]) if len(finish_candidates) and final<=minimum+15 else None
 pre_mask=finite&(t<(t[onset] if onset is not None else t[-1]));post_mask=finite&(t>=t[finish]+.3) if finish is not None else np.zeros(len(t),bool)
 lstm=v('lstm_mm');prefirst=np.flatnonzero(pre_mask)
 pre_drift=float(np.ptp(lstm[pre_mask])) if pre_mask.sum()>=5 else np.nan
 peaks,_=find_peaks(-smooth,prominence=max(30,amplitude*.25),distance=max(1,int(fps*.4)))
 perf=json.loads((p/'recording_performance.json').read_text()) if (p/'recording_performance.json').exists() else {}
 videos={}
 for name in ['camera_raw.avi','camera_overlay_native.avi','prediction_curves.avi','synchronized_demo_hd.avi']:
  cap=cv2.VideoCapture(str(p/name));videos[name]=dict(opened=cap.isOpened(),frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),fps=cap.get(cv2.CAP_PROP_FPS),width=int(cap.get(3)),height=int(cap.get(4)));cap.release()
 counts=[x['frames'] for x in videos.values()]
 rec=dict(number=number,trial=p.name,time=p.name.split('_')[2],size_mm=m['size_cm']*10,duration_s=float(t[-1]-t[0]),frames=len(t),capture_fps=float(fps),marker_valid_pct=float(valid.mean()*100),prediction_valid_pct=float(finite.mean()*100),initial_distance_mm=initial,final_distance_mm=final,min_distance_mm=minimum,onset_s=float(t[onset]) if onset is not None else None,movement_duration_s=float(t[finish]-t[onset]) if finish is not None and onset is not None else None,pre_rest_duration_s=float(t[onset]-t[0]) if onset is not None else None,lstm_pre_rest_range_mm=pre_drift,lstm_peak_mm=float(np.nanmax(lstm)) if finite.any() else None,lstm_final_mm=float(np.nanmedian(lstm[t>=t[-1]-.5])) if finite.any() else None,lstm_post_rest_sd_mm=float(np.std(lstm[post_mask])) if post_mask.sum()>4 else None,function_peak_mm=float(np.nanmax(v('function_mm'))),mlp_peak_mm=float(np.nanmax(v('mlp_mm'))),tracking_gaps_over_100ms=int((dt>.1).sum()),approach_cycles_detected=len(peaks),encoder_skipped_frames=perf.get('encoder_skipped_input_frames',0),video_frames_match=len(set(counts))==1 and min(counts)>0,video_duration_s=counts[0]/videos['camera_raw.avi']['fps'] if videos['camera_raw.avi']['fps'] else 0)
 records.append(rec);loaded.append((p,t,d,rows,rec,videos))
 print(number,rec['time'],'duration',round(rec['duration_s'],2),'movement',round(rec['movement_duration_s'],2) if rec['movement_duration_s'] is not None else None,'distance',round(initial),round(final),'valid',round(rec['marker_valid_pct'],1),'preDrift',round(pre_drift,2),'LSTMpeak',round(rec['lstm_peak_mm'],1),'gaps',rec['tracking_gaps_over_100ms'],'video',rec['video_frames_match'],flush=True)
with (out/'per_trial_review.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
fig,axes=plt.subplots(4,4,figsize=(20,14),dpi=160)
for ax,(p,t,d,rows,r,vid) in zip(axes.flat,loaded):
 for k,label,color in [('function_mm','Function','#2786bd'),('mlp_mm','MLP','#ed9727'),('lstm_mm','LSTM','#389b51')]:ax.plot(t,[float(x[k]) for x in rows],color=color,lw=1,label=label)
 ax.set_title(f"{r['number']:02d} | {r['time'][:2]}:{r['time'][2:4]}:{r['time'][4:]} | {r['duration_s']:.1f} s",fontsize=11)
 ax.set_ylim(-10,130);ax.set_xlabel('Time (s)',fontsize=9);ax.grid(alpha=.18)
 if r['onset_s'] is not None:ax.axvline(r['onset_s'],c='#888888',ls=':',lw=.8)
 if r['movement_duration_s'] is not None:ax.axvline(r['onset_s']+r['movement_duration_s'],c='#888888',ls=':',lw=.8)
for ax in axes[:,0]:ax.set_ylabel('Predicted aperture (mm)')
handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False);fig.tight_layout(rect=(0,0,1,.97));fig.savefig(out/'all16_prediction_overview.png');plt.close(fig)
fig,axes=plt.subplots(4,4,figsize=(20,13),dpi=140)
for ax,(_,t,d,rows,r,vid) in zip(axes.flat,loaded):ax.plot(t,d,color='#347f9d',lw=1.2);ax.set_title(f"{r['number']:02d} | {r['time']} | {r['initial_distance_mm']:.0f} -> {r['final_distance_mm']:.0f} mm",fontsize=10);ax.set_ylim(0,370);ax.grid(alpha=.2);ax.set_xlabel('Time (s)',fontsize=9)
fig.tight_layout();fig.savefig(out/'all16_distance_overview.png');plt.close(fig)
(out/'video_integrity.json').write_text(json.dumps({p.name:vid for p,t,d,rows,r,vid in loaded},indent=2),encoding='utf-8')
