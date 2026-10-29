"""Export explicitly selected illustrative closing-phase examples from saved predictions."""
import os,csv,json
from pathlib import Path
os.environ['MPLCONFIGDIR']=str(Path(__file__).parent/'.mplconfig')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1]/'Data/1029_D1';OUT=B/'report_prediction_examples'/'closing_phase_selected';OUT.mkdir(parents=True,exist_ok=True)
CASES=[('20mm_red','Train','grasp_051','20mm_Train_grasp_051'),('40mm_black','Test','grasp_092','40mm_Test_grasp_092'),('20mm_red','Test','grasp_094','20mm_Test_grasp_094_alternative')]
KEYS=['function_mm','mlp_5f_mm','lstm_full_mm'];LABELS=['Function','MLP 5f','LSTM Full'];COLORS=['#1f77b4','#ff7f0e','#2ca02c'];report=[]
for folder,split,trial,name in CASES:
 run=sorted(p for p in (B/folder/'Result').glob('l0_normalized_function_mlp5_lstmfull_400ep_*') if (p/'per_trial_metrics.csv').exists())[-1];source=run/split/'predictions'/f'{trial}.csv'
 with source.open(encoding='utf-8-sig',newline='') as f:r=list(csv.DictReader(f))
 v=lambda k:np.array([float(x[k]) if x[k] else np.nan for x in r]);t=v('time_s');y=v('actual_aperture_mm');common=v('common_valid')==1
 meta=json.loads((B/folder/split/trial/'segment.json').read_text());end=meta['closure_confirmed_row']-meta['start_row'];peak=int(np.argmax(y[:end+1]));idx=np.arange(peak,end+1);idx=idx[common[idx]]
 errors={label:float(np.mean(abs(v(key)[idx]-y[idx]))) for label,key in zip(LABELS,KEYS)}
 assert errors['Function']>errors['MLP 5f']>errors['LSTM Full']
 fig,ax=plt.subplots(figsize=(4.6,5.6),facecolor='white');fig.subplots_adjust(left=.18,right=.97,bottom=.12,top=.98)
 ax.plot(t,y,color='black',lw=1.7,label='Measured')
 for key,label,color in zip(KEYS,LABELS,COLORS):
  p=v(key).copy();p[~common]=np.nan;ax.plot(t,p,color=color,lw=1.5,label=label)
 ax.set_xlabel('Time (s)',fontsize=12);ax.set_ylabel('Finger aperture (mm)',fontsize=12);ax.grid(alpha=.2);ax.legend(loc='upper right',fontsize=9);ax.tick_params(labelsize=10)
 for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=300,facecolor='white')
 plt.close(fig)
 with (run/'per_trial_metrics.csv').open(encoding='utf-8-sig') as f:overall={x['model']:float(x['mae_mm']) for x in csv.DictReader(f) if x['split']==split and x['trial']==trial}
 assert overall['Function']>max(overall['MLP 5f'],overall['LSTM Full'])
 report.append(dict(object_directory=folder,split=split,trial=trial,prediction_csv=str(source),closing_interval_s=[float(t[peak]),float(t[end])],closing_samples=len(idx),closing_mae_mm=errors,overall_mae_mm=overall,figure=str(OUT/f'{name}.png')))
(OUT/'selection_manifest.json').write_text(json.dumps(dict(selection='Illustrative cases deliberately selected for closing-phase MAE Function > MLP 5f > LSTM Full, and Function overall MAE worse than both networks. Not random or median-representative examples. Train/Test source disclosed here and must be disclosed when presented.',phase_definition='Measured aperture maximum up to annotated closure confirmation; common valid frames only.',interpretation='Shows behaviour of these selected trials, not a controlled window-size experiment or proof of a universal ordering.',data_processing='Saved predictions unchanged; actual timestamps, mm outputs; no smoothing, temporal shift or prediction adjustment.',cases=report),indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
