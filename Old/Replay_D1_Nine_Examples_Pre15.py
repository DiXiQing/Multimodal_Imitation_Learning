"""Restore five authentic preceding frames and replay the original held-out fold models."""
from pathlib import Path
import csv,json,shutil,sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'Old'))
import Train_Joint_Three_Models_SmoothSG5_Recent5_CurrentV_CV5 as base

def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
 torch.set_num_threads(4)
 parent=ROOT/'Report/Two_Participant_Comparison/D1_Prediction_Examples'
 restored=parent/'PreAction15'
 out=parent/'PreAction15_RePredicted';out.mkdir(exist_ok=True)
 manifest=json.loads((restored/'restoration_manifest.json').read_text())
 selection=read(parent/'source_selection.csv')
 fig,axes=plt.subplots(3,3,figsize=(14,7),dpi=220);handles=[];audit=[]
 for c,size in enumerate([20,30,40]):
  for row,rec in enumerate(r for r in selection if int(r['size_mm'])==size):
   name=rec['trial'];original=ROOT/'Data/1101_D'/f'{size}mm'/name
   source=Path(next(r['original_source'] for r in manifest if r['trial']==name))
   raw=read(source);clip=read(original/'camera.csv')
   i=next(i for i,r in enumerate(raw) if r['frame']==clip[0]['frame'] and abs(float(r['t_camera'])-float(clip[0]['t_camera']))<1e-5)
   assert i>=5
   rows=raw[i-5:i]+clip
   keys=list(dict.fromkeys(k for r in rows for k in r))
   rows=[{k:r.get(k,'') for k in keys} for r in rows]
   folder=out/'input_trials'/f'{size}mm'/name;folder.mkdir(parents=True,exist_ok=True)
   base.write_csv(folder/'camera.csv',rows)
   shutil.copy2(original/'settings.json',folder/'settings.json')
   meta=json.loads((original/'segment.json').read_text(encoding='utf-8-sig'))
   meta.update(pre_action_frames=15,frames=len(rows),restored_preceding_frames=5)
   meta['start_row']-=5;meta['source_start_frame']=int(rows[0]['frame'])
   base.write_json(folder/'segment.json',meta)
   g=base.load_trial(folder);g['size_cm']=size//10;g['use_velocity']=True
   fold=Path(rec['source_csv']).parents[2]
   function=json.loads((fold/'function.json').read_text())
   scaling=function['velocity_scaling'];g['velocity_scaled']=(g['velocity_mm_s']-scaling['mean_mm_s'])/scaling['std_mm_s']
   pred={key:np.full(len(g['d']),np.nan) for key in ['function_mm','mlp_5f_mm','lstm_full_mm']}
   pred['function_mm'][g['common']]=base.design(g,function['train_x_min'],function['train_x_range'])@np.array(function['coefficients'])
   torch.manual_seed(20261029+100+int(fold.name[-2:]))
   with torch.inference_mode():
    for label,key,file,cls in [('MLP 5f','mlp_5f_mm','mlp5.pth',base.JointMLP),('LSTM Full','lstm_full_mm','lstm_full.pth',base.JointLSTM)]:
     checkpoint=torch.load(fold/file,map_location='cpu',weights_only=False)
     model=cls(True);model.load_state_dict(checkpoint['state_dict']);model.eval()
     for ids,x in base.batches(g,label,'cpu'):
      values=base.predict_mlp_mc(model,x) if label=='MLP 5f' else model(x).reshape(-1)
      pred[key][ids]=values.numpy()*checkpoint['std_mm']+checkpoint['mean_mm']
   plotted=[dict(frame=int(g['frame'][j]),time_s=float(g['t'][j]),actual_mm=float(g['y'][j]),**{key:float(value[j]) for key,value in pred.items()}) for j in range(len(rows))]
   base.write_csv(out/f'{size}mm_{name}_predictions.csv',plotted)
   ax=axes[row,c]
   for key,label,color in [('actual_mm','Measured','#222222'),('function_mm','Function','#3485b8'),('mlp_5f_mm','MLP 5f','#f29d38'),('lstm_full_mm','LSTM Full','#519e58')]:
    line,=ax.plot(g['t'],[r[key] for r in plotted],label=label,color=color,lw=1.2)
    if row==0 and c==0:handles.append(line)
   ax.set(xlim=(0,4),ylim=(0,125),title=f'{size} mm',xlabel='Time (s)');ax.set_xticks(np.arange(0,4.01,.5));ax.set_yticks(np.arange(0,126,25));ax.tick_params(labelsize=9);ax.grid(alpha=.18)
   if c==0:ax.set_ylabel('d (mm)')
   else:ax.tick_params(labelleft=False)
   first=int(np.flatnonzero(np.isfinite(pred['function_mm']))[0])
   audit.append(dict(trial=name,fold=str(fold),pre_action_frames=15,first_prediction_index_zero_based=first,first_prediction_time_s=float(g['t'][first]),model_retrained=False,original_preprocessing_preserved=True))
   print(name,fold.name,'first prediction frame',first+1,flush=True)
 fig.legend(handles=handles,loc='upper right',bbox_to_anchor=(.99,1),ncol=4,frameon=False,fontsize=10);fig.tight_layout(rect=(0,0,1,.96),h_pad=1.3)
 for ext in ['png','svg','pdf']:fig.savefig(out/f'D1_nine_examples_pre15_repredicted_0_4s.{ext}')
 base.write_json(out/'replay_manifest.json',audit)
 print(out,flush=True)
if __name__=='__main__':main()

