"""Highlight main closing intervals in the three user-selected examples."""
import os,csv,json,hashlib
from pathlib import Path
os.environ['MPLCONFIGDIR']=str(Path(__file__).parent/'.mplconfig')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1]/'Data/1029_D1';O=B/'report_prediction_examples'/'highlighted_closing';O.mkdir(parents=True,exist_ok=True)
CASES=[('20mm_red','Train','grasp_051',True,20),('30mm_blue','Test','grasp_100',False,30),('40mm_black','Train','grasp_103',False,40)]
report=[]
for folder,split,trial,edited,size in CASES:
 run=sorted(p for p in (B/folder/'Result').glob('l0_normalized_function_mlp5_lstmfull_400ep_*') if (p/'per_trial_metrics.csv').exists())[-1];file=run/split/'predictions'/f'{trial}.csv';before=hashlib.sha256(file.read_bytes()).hexdigest()
 with file.open(encoding='utf-8-sig',newline='') as f:r=list(csv.DictReader(f))
 v=lambda k:np.array([float(x[k]) if x[k] else np.nan for x in r]);t=v('time_s');y=v('actual_aperture_mm');common=v('common_valid')==1
 if edited:
  with (B/folder/split/trial/'camera.csv').open(encoding='utf-8-sig',newline='') as f:raw={int(x['frame']):x for x in csv.DictReader(f)}
  y=np.array([float(raw[int(x['source_frame'])]['finger_width_mm']) for x in r])
 meta=json.loads((B/folder/split/trial/'segment.json').read_text());closure=meta['closure_confirmed_row']-meta['start_row'];peak=int(np.nanargmax(y[:closure+1]));hold=float(np.nanmedian(y[-10:]));amplitude=float(y[peak]-hold)
 assert amplitude>0
 high=hold+.9*amplitude;low=hold+.1*amplitude
 def crossing(threshold):
  # Use recorded samples for thresholds; interpolate boundary time only.
  for i in range(peak+1,len(y)):
   if np.isfinite(y[i-1:i+1]).all() and y[i]<=threshold<y[i-1]:
    return float(t[i-1]+(threshold-y[i-1])/(y[i]-y[i-1])*(t[i]-t[i-1]))
  raise ValueError(f'No descending threshold crossing: {folder}/{trial}')
 start=crossing(high);end=crossing(low);assert start<end
 fig,ax=plt.subplots(figsize=(4.6,5.6),facecolor='white');fig.subplots_adjust(left=.18,right=.97,bottom=.12,top=.98)
 ax.axvspan(start,end,facecolor='#FFD966',alpha=.32,zorder=0)
 ax.plot(t,y,color='black',lw=1.7,label='Reference (edited)' if edited else 'Measured',zorder=3)
 for key,label,color in zip(['function_mm','mlp_5f_mm','lstm_full_mm'],['Function','MLP 5f','LSTM Full'],['#1f77b4','#ff7f0e','#2ca02c']):
  pred=v(key);pred[~common]=np.nan;ax.plot(t,pred,color=color,lw=1.5,label=label,zorder=3)
 ax.set_xlabel('Time (s)',fontsize=12);ax.set_ylabel('Finger aperture (mm)',fontsize=12);ax.grid(alpha=.2);ax.set_axisbelow(True);ax.tick_params(labelsize=10)
 ax.legend(loc='upper right',fontsize=7.5 if edited else 9,handlelength=1.5 if edited else 2,labelspacing=.3 if edited else .5,borderpad=.35 if edited else .4)
 name=f'{size}mm_{split}_{trial}_closing_highlight'
 for ext in ['png','pdf','svg']:fig.savefig(O/f'{name}.{ext}',dpi=300,facecolor='white')
 plt.close(fig)
 assert hashlib.sha256(file.read_bytes()).hexdigest()==before
 report.append(dict(object_mm=size,split=split,trial=trial,source_prediction_csv=str(file),reference_manually_edited=edited,highlight_start_s=start,highlight_end_s=end,high_threshold_mm=high,low_threshold_mm=low,holding_reference_mm=hold,figure=str(O/f'{name}.png')))
(O/'highlight_manifest.json').write_text(json.dumps(dict(definition='Main closing region: reference aperture decreases from hold+90%*(peak-hold) to hold+10%*(peak-hold); hold = last 10-frame median. Only boundary timestamps interpolated; all plotted measurements/predictions unchanged, no smoothing.',cases=report),indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
