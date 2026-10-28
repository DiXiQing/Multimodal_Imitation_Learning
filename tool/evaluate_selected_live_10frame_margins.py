"""Re-score ten selected recordings with ten-frame rest/hold margins; no re-inference."""
from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'New/Result/RealTime/four_model_live_20261028_163345_674439/selected_standard5_by_distance'
OUT=SOURCE/'trimmed_10frame_evaluation'
KEYS=['predicted_function_px','predicted_mlp_px','predicted_lstm_window5_px','predicted_lstm_px']
LABELS=['Function','MLP 5f','LSTM 5f','LSTM Full']
COLORS=['#3485b8','#e59a32','#9565b0','#519e58']

def score(d,key):
 y=d.finger_width.to_numpy();p=d[key].to_numpy();e=p-y;den=np.sum((y-y.mean())**2)
 return dict(frames=len(d),mae=float(np.mean(abs(e))),rmse=float(np.sqrt(np.mean(e**2))),r2=float(1-np.sum(e**2)/den) if den>1e-12 else None)

def main():
 if OUT.exists():raise ValueError(f'Output exists: {OUT}')
 OUT.mkdir()
 parts=[];bounds=[];metrics=[];sources={}
 paths=sorted(SOURCE.glob('near/trial_*/predictions.csv'))+sorted(SOURCE.glob('far/trial_*/predictions.csv'))
 assert len(paths)==10
 fig,axes=plt.subplots(2,5,figsize=(20,7),sharey=True)
 for ax,p in zip(axes.flat,paths):
  sources[p]=hashlib.sha256(p.read_bytes()).hexdigest();d=pd.read_csv(p);a=d.finger_width.to_numpy();sm=pd.Series(a).rolling(3,center=True,min_periods=1).median().to_numpy();x=d.hand_object_signed_distance.to_numpy()
  base=float(np.median(sm[:10]));peak=int(np.argmax(sm));onset=None
  for j in range(10,peak):
   if np.all(sm[j:j+3]>base+5):onset=j;break
  if onset is None:raise ValueError(f'No opening onset: {p}')
  final=float(np.median(sm[-20:]));stable=None
  for j in range(peak+1,len(d)-9):
   if np.max(abs(sm[j:j+5]-final))<=3 and np.median(abs(x[j:j+5]))<20:stable=j;break
  if stable is None:raise ValueError(f'No stable hold: {p}')
  start=onset-10;end=stable+9
  assert start>=0 and end<len(d)
  condition=p.parent.parent.name;trial=p.parent.name
  crop=d.iloc[start:end+1].copy();crop.insert(0,'original_elapsed_s',crop.elapsed_s);crop['elapsed_s']=crop.t_camera-crop.t_camera.iloc[0]
  valid=crop.valid.astype(str).str.lower().isin(['true','1']) & crop.common_valid.astype(str).str.lower().isin(['true','1']) & np.isfinite(crop[['finger_width',*KEYS]]).all(axis=1)
  common=crop.loc[valid].copy();common['condition']=condition;common['trial']=trial;parts.append(common)
  dest=OUT/condition/trial;dest.mkdir(parents=True);crop.to_csv(dest/'predictions_trimmed.csv',index=False,encoding='utf-8-sig')
  bounds.append(dict(condition=condition,trial=trial,original_frames=len(d),retained_frames=len(crop),common_frames=len(common),start_row=start,onset_row=onset,stable_row=stable,end_row=end,pre_onset_frames=onset-start,hold_frames=end-stable+1,original_start_s=float(d.elapsed_s.iloc[start]),original_end_s=float(d.elapsed_s.iloc[end]),retained_duration_s=float(crop.elapsed_s.iloc[-1]),baseline_aperture_px=base,hold_aperture_px=final))
  for key,label in zip(KEYS,LABELS):metrics.append(dict(condition=condition,trial=trial,model=label,**score(common,key)))
  ax.plot(crop.elapsed_s,crop.finger_width,color='black',lw=1.7,label='Measured')
  for key,label,c in zip(KEYS,LABELS,COLORS):ax.plot(crop.elapsed_s,crop[key],color=c,lw=1.1,label=label)
  ax.axvline(float(d.t_camera.iloc[onset]-crop.t_camera.iloc[0]),color='gray',ls=':',lw=.7)
  ax.axvline(float(d.t_camera.iloc[stable]-crop.t_camera.iloc[0]),color='gray',ls=':',lw=.7)
  ax.set_title(f'{condition} | {trial[15:21]}',fontsize=10);ax.set_xlabel('Retained time (s)');ax.grid(alpha=.18)
 axes[0,0].set_ylabel('Aperture (px)');axes[1,0].set_ylabel('Aperture (px)')
 fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper center',ncol=5,frameon=False);fig.tight_layout(rect=(0,0,1,.95))
 for ext in ['png','pdf']:fig.savefig(OUT/f'trimmed_predictions.{ext}',dpi=180)
 plt.close(fig)
 all_data=pd.concat(parts,ignore_index=True);summary=[]
 for group in ['near','far','Overall']:
  d=all_data if group=='Overall' else all_data[all_data.condition==group]
  for key,label in zip(KEYS,LABELS):summary.append(dict(condition=group,model=label,trials=d.trial.nunique(),**score(d,key)))
 pd.DataFrame(bounds).to_csv(OUT/'trim_boundaries.csv',index=False,encoding='utf-8-sig');pd.DataFrame(metrics).to_csv(OUT/'trial_metrics.csv',index=False,encoding='utf-8-sig');pd.DataFrame(summary).to_csv(OUT/'summary_metrics.csv',index=False,encoding='utf-8-sig')
 fig,axes=plt.subplots(3,3,figsize=(13,10))
 for row,group in zip(axes,['near','far','Overall']):
  data=[r for r in summary if r['condition']==group]
  for ax,key,title in zip(row,['mae','rmse','r2'],['MAE (px)','RMSE (px)','R²']):
   bars=ax.bar(LABELS,[r[key] for r in data],color=COLORS);ax.bar_label(bars,fmt='%.3f' if key=='r2' else '%.2f',padding=3);ax.set_title(group+' | '+title);ax.tick_params(axis='x',labelsize=8);ax.margins(y=.2);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 fig.tight_layout()
 for ext in ['png','pdf']:fig.savefig(OUT/f'metrics_by_distance.{ext}',dpi=180)
 plt.close(fig)
 (OUT/'method.json').write_text(json.dumps(dict(source=str(SOURCE),opening_onset='First three consecutive median-filtered aperture samples > initial ten-frame median +5 px; search before maximum aperture.',stable_hold='After peak: five consecutive median-filtered aperture samples within 3 px of final twenty-frame median and median absolute signed distance <20 px.',margins='Exactly ten source rows before opening onset and ten hold rows including first stable row.',smoothing='Three-point centered median for boundary detection only; saved/scored values unchanged.',scoring='Pooled common-valid finite frames, separately near/far and combined; all four models use identical rows.',predictions='Existing real-time predictions retained, including original LSTM history; no inference reset, re-training, clipping or prediction edits.',selection='Ten previously selected representative trials; not an all-trial evaluation.',source_sha256={str(p):h for p,h in sources.items()}),indent=2),encoding='utf-8')
 for p,h in sources.items():assert hashlib.sha256(p.read_bytes()).hexdigest()==h
 print(pd.DataFrame(bounds)[['condition','trial','original_frames','retained_frames','pre_onset_frames','hold_frames']].to_string(index=False))
 print(pd.DataFrame(summary).round(4).to_string(index=False))
if __name__=='__main__':main()
