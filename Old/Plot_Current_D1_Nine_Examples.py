from pathlib import Path
import csv,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'Program'))
import Train_D1_Joint_Three_Models_LinearFit5_CV5 as base
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
RESULT=ROOT/'Data/1101_D_PreAction15/Result/Joint_Three_Models_LinearFit5_CurrentV_CV5_20261123_153838'
OUT=ROOT/'Report/Two_Participant_Comparison/Current_D1_Prediction_Examples'
def read(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def main():
 torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
 conditions=read(ROOT/'Report/Two_Participant_Comparison/Replacement_3p0s/current600_conditions.csv');metrics=read(RESULT/'oof_per_trial_metrics.csv')
 index={(r['size_cm'],r['trial'],r['model']):r for r in metrics};selected=[]
 for size in [20,30,40]:
  eligible=[]
  for r in conditions:
   if int(r['size_mm'])!=size:continue
   name=r['trial'];key=(str(size//10),name,'LSTM Full')
   if key not in index:continue
   current=ROOT/'Data/1101_D'/f'{size}mm'/name;prepared=ROOT/'Data/1101_D_PreAction15'/f'{size}mm'/name
   a=read(current/'camera.csv');b=read(prepared/'camera.csv')
   if len(b)<len(a):continue
   # Exclude time-scaled or altered measurements; restored five-frame prefix is allowed.
   b=b[-len(a):]
   if any(not np.allclose([float(x[k]) for x in a],[float(x[k]) for x in b],equal_nan=True,rtol=0,atol=1e-6) for k in ['t_camera','finger_width_mm','index_mcp_object_distance_mm']):continue
   model_metrics={m:index[(str(size//10),name,m)] for m in ['Function','MLP 5f','LSTM Full']}
   best=float(model_metrics['LSTM Full']['mae_mm'])<min(float(model_metrics[m]['mae_mm']) for m in ['Function','MLP 5f'])
   if not best:continue
   eligible.append(dict(**r,fold=int(model_metrics['LSTM Full']['fold']),prior_lstm_mae=float(model_metrics['LSTM Full']['mae_mm']),folder=current))
  for label,lo,hi in [('Short',1.,1.5),('Medium',1.5,2.25),('Long',2.25,3.00001)]:
   pool=[r for r in eligible if lo<=float(r['action_duration_s'])<hi]
   if not pool:raise ValueError(f'No candidates {size} {label}')
   median=float(np.median([r['prior_lstm_mae'] for r in pool]))
   # Select a typical LSTM-leading case within each duration band, not the minimum error.
   pool.sort(key=lambda r:abs(r['prior_lstm_mae']-median))
   choice=pool[0];choice['duration_group']=label;selected.append(choice)
 fig,axes=plt.subplots(3,3,figsize=(16,8),dpi=220);handles=[];audit=[];newmetrics=[]
 for c,size in enumerate([20,30,40]):
  for row,rec in enumerate(r for r in selected if int(r['size_mm'])==size):
   g=base.load_trial(rec['folder']);g['size_cm']=size//10;g['use_velocity']=True
   fold=RESULT/f"fold_{rec['fold']:02d}";fn=json.loads((fold/'function.json').read_text());scale=fn['velocity_scaling'];g['velocity_scaled']=(g['velocity_mm_s']-scale['mean_mm_s'])/scale['std_mm_s'];ids=g['common']
   pred={key:np.full(len(g['d']),np.nan) for key in ['function_mm','mlp_5f_mm','lstm_full_mm']}
   pred['function_mm'][ids]=base.design(g,fn['train_x_min'],fn['train_x_range'])@np.array(fn['coefficients'])
   torch.manual_seed(20261029+100+rec['fold'])
   with torch.inference_mode():
    for label,key,file,cls in [('MLP 5f','mlp_5f_mm','mlp5.pth',base.JointMLP),('LSTM Full','lstm_full_mm','lstm_full.pth',base.JointLSTM)]:
     ck=torch.load(fold/file,map_location='cpu',weights_only=False);model=cls(True);model.load_state_dict(ck['state_dict']);model.eval()
     for idx,x in base.batches(g,label,'cpu'):
      v=base.predict_mlp_mc(model,x) if label=='MLP 5f' else model(x).reshape(-1)
      pred[key][idx]=v.numpy()*ck['std_mm']+ck['mean_mm']
   rows=[dict(frame=int(g['frame'][i]),time_s=float(g['t'][i]),actual_mm=float(g['y'][i]),**{k:float(v[i]) for k,v in pred.items()}) for i in range(len(g['d']))]
   base.write_csv(OUT/'predictions'/f'{size}mm'/f"{rec['trial']}.csv",rows)
   for key,label in [('function_mm','Function'),('mlp_5f_mm','MLP 5f'),('lstm_full_mm','LSTM Full')]:newmetrics.append(dict(size_mm=size,trial=rec['trial'],model=label,**base.scores(g['y'][ids],pred[key][ids])))
   ax=axes[row,c]
   for key,label,color in [('actual_mm','Measured','#222222'),('function_mm','Function','#2786bd'),('mlp_5f_mm','MLP 5f','#f09527'),('lstm_full_mm','LSTM Full','#389b51')]:
    y=np.array([r[key] for r in rows])
    if key!='actual_mm':y[~np.isin(np.arange(len(y)),ids)]=np.nan
    line,=ax.plot(g['t'],y,color=color,label=label,lw=1.4)
    if row==0 and c==0:handles.append(line)
   ax.set_xlim(0,4);ax.set_ylim(0,125);ax.set_xticks(np.arange(0,4.01,.5));ax.set_yticks(np.arange(0,126,25));ax.set_title(f'{size} mm',fontsize=16,pad=9);ax.set_xlabel('Time (s)',fontsize=12);ax.tick_params(labelsize=11);ax.grid(alpha=.22);ax.set_axisbelow(True)
   if c==0:ax.set_ylabel('Finger aperture d(mm)',fontsize=12)
   else:ax.tick_params(labelleft=False)
   audit.append(dict(size_mm=size,duration_group=rec['duration_group'],trial=rec['trial'],action_duration_s=float(rec['action_duration_s']),clip_duration_s=float(g['t'][-1]),fold=rec['fold'],source=str(rec['folder']),selection='Near-median old OOF LSTM MAE among LSTM-leading unchanged current trials in duration band; current CSV replayed',retrained=False))
   print(size,rec['duration_group'],rec['trial'],float(g['t'][-1]),flush=True)
 fig.legend(handles=handles,loc='upper right',bbox_to_anchor=(.993,1.005),ncol=4,frameon=False,fontsize=13);fig.tight_layout(rect=(0,0,1,.955),h_pad=1.4,w_pad=1.4)
 for ext in ['png','svg','pdf']:fig.savefig(OUT/f'D1_current_trials_LinearFit5_nine_examples.{ext}')
 base.write_csv(OUT/'selected_trials.csv',audit);base.write_csv(OUT/'selected_trial_metrics.csv',newmetrics)
if __name__=='__main__':main()
