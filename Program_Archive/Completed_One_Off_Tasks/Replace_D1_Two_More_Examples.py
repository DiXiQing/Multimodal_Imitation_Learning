from pathlib import Path
import csv,json,sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'Program'))
import Train_D1_Joint_Three_Models_LinearFit5_CV5 as base
OUT=ROOT/'Report/Two_Participant_Comparison/Current_D1_Prediction_Examples'
RESULT=ROOT/'Data/1101_D_PreAction15/Result/Joint_Three_Models_LinearFit5_CurrentV_CV5_20261123_153838'
def read(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def replay(folder,size,fold):
 g=base.load_trial(folder);g['size_cm']=size//10;g['use_velocity']=True
 modeldir=RESULT/f'fold_{fold:02d}';fn=json.loads((modeldir/'function.json').read_text());sc=fn['velocity_scaling'];g['velocity_scaled']=(g['velocity_mm_s']-sc['mean_mm_s'])/sc['std_mm_s'];ids=g['common']
 pred={key:np.full(len(g['d']),np.nan) for key in ['function_mm','mlp_5f_mm','lstm_full_mm']};pred['function_mm'][ids]=base.design(g,fn['train_x_min'],fn['train_x_range'])@np.array(fn['coefficients']);torch.manual_seed(20261029+100+fold)
 with torch.inference_mode():
  for label,key,file,cls in [('MLP 5f','mlp_5f_mm','mlp5.pth',base.JointMLP),('LSTM Full','lstm_full_mm','lstm_full.pth',base.JointLSTM)]:
   ck=torch.load(modeldir/file,map_location='cpu',weights_only=False);model=cls(True);model.load_state_dict(ck['state_dict']);model.eval()
   for ix,x in base.batches(g,label,'cpu'):
    v=base.predict_mlp_mc(model,x) if label=='MLP 5f' else model(x).reshape(-1)
    pred[key][ix]=v.numpy()*ck['std_mm']+ck['mean_mm']
 rows=[dict(time_s=float(g['t'][i]),actual_mm=float(g['y'][i]),**{k:float(v[i]) for k,v in pred.items()}) for i in range(len(g['d']))]
 scores={k:base.scores(g['y'][ids],v[ids])['mae_mm'] for k,v in pred.items()}
 return rows,scores

def main():
 torch.set_num_threads(2);selection=read(OUT/'D1_keep7_replace2_sources.csv');metrics=read(RESULT/'oof_per_trial_metrics.csv');conditions=read(ROOT/'Report/Two_Participant_Comparison/Replacement_3p0s/current600_conditions.csv');used={r['trial'] for r in selection}
 details=[]
 for size,target_group,lo,hi in [(30,'Medium',1.5,2.25),(20,'Long',2.15,3.0)]:
  candidates=[]
  for m in metrics:
   if m['model']!='LSTM Full' or int(m['size_cm'])*10!=size or m['trial'] in used:continue
   r=next((r for r in conditions if int(r['size_mm'])==size and r['trial']==m['trial']),None)
   if not r or not lo<=float(r['action_duration_s'])<=hi:continue
   folder=ROOT/'Data/1101_D'/f'{size}mm'/m['trial'];a=read(folder/'camera.csv');b=read(ROOT/'Data/1101_D_PreAction15'/f'{size}mm'/m['trial']/'camera.csv')
   if float(a[-1]['t_camera'])-float(a[0]['t_camera'])>3:continue
   if len(b)<len(a):continue
   b=b[-len(a):]
   if any(not np.allclose([float(x[k]) for x in a],[float(x[k]) for x in b],equal_nan=True,rtol=0,atol=1e-6) for k in ['t_camera','finger_width_mm','index_mcp_object_distance_mm']):continue
   candidates.append((float(m['mae_mm']),folder,int(m['fold'])))
  evaluated=[]
  for _,folder,fold in sorted(candidates)[:20]:
   rows,scores=replay(folder,size,fold)
   if scores['lstm_full_mm']<min(scores['mlp_5f_mm'],scores['function_mm']):evaluated.append((scores['lstm_full_mm'],folder,fold,rows,scores))
  if not evaluated:raise ValueError(f'No LSTM-leading candidate for {size}')
  _,folder,fold,rows,scores=min(evaluated,key=lambda r:r[0]);source=OUT/'predictions'/f'{size}mm'/f'{folder.name}_replacement.csv';base.write_csv(source,rows)
  target=next(r for r in selection if int(r['size_mm'])==size and r['duration_group']==target_group);target.update(trial=folder.name,source=str(source),status='Performance-selected LSTM-leading example; current CSV, original held-out fold; no retraining')
  details.append(dict(size_mm=size,trial=folder.name,fold=fold,**scores));print(details[-1],flush=True)
 fig,axes=plt.subplots(3,3,figsize=(16,8),dpi=220);handles=[]
 for c,size in enumerate([20,30,40]):
  for row,group in enumerate(['Short','Medium','Long']):
   rec=next(r for r in selection if int(r['size_mm'])==size and r['duration_group']==group);data=read(Path(rec['source']));assert float(data[-1]['time_s'])<=3
   ax=axes[row,c]
   for key,label,color in [('actual_mm','Measured','#222222'),('function_mm','Function','#2786bd'),('mlp_5f_mm','MLP 5f','#f09527'),('lstm_full_mm','LSTM Full','#389b51')]:
    y=np.array([float(r[key]) for r in data])
    if key!='actual_mm' and 'common_valid' in data[0]:y[np.array([r['common_valid']!='1' for r in data])]=np.nan
    line,=ax.plot([float(r['time_s']) for r in data],y,label=label,color=color,lw=1.4)
    if row==0 and c==0:handles.append(line)
   ax.set_xlim(0,3);ax.set_ylim(0,125);ax.set_xticks(np.arange(0,3.01,.5));ax.set_yticks(np.arange(0,126,25));ax.set_title(f'{size} mm',fontsize=16,pad=9);ax.set_xlabel('Time (s)',fontsize=12);ax.tick_params(labelsize=11);ax.grid(alpha=.22);ax.set_axisbelow(True)
   if c==0:ax.set_ylabel('Finger aperture d(mm)',fontsize=12)
   else:ax.tick_params(labelleft=False)
 fig.legend(handles=handles,loc='upper right',bbox_to_anchor=(.993,1.005),ncol=4,frameon=False,fontsize=13);fig.tight_layout(rect=(0,0,1,.955),h_pad=1.4,w_pad=1.4)
 for ext in ['png','svg','pdf']:fig.savefig(OUT/f'D1_updated_four_examples_time3s.{ext}')
 base.write_csv(OUT/'D1_updated_four_examples_sources.csv',selection);base.write_csv(OUT/'second_two_replacement_metrics.csv',details)
if __name__=='__main__':main()
