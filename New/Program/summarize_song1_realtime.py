"""Summarize six recorded realtime grasps; no model training."""
import csv,json
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import summarize_d1_validation as report
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'Data/TopView/Inference/run_20261018_153832_802120/trial_001'
class MLP(nn.Module):
 def __init__(self,n,h):
  super().__init__();self.net=nn.Sequential(nn.Linear(n,h),nn.ReLU(),nn.Dropout(.1),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,1))
 def forward(self,x):return self.net(x).squeeze(-1)
def main():
 out=ROOT/'New/Result'/f'SONG1_realtime_summary_{datetime.now():%Y%m%d_%H%M%S}';out.mkdir()
 manifest=json.loads((SOURCE/'grasp_segments/segments_manifest.json').read_text());config=json.loads((SOURCE/'config.json').read_text())
 targets={'blue':'Blue 30 mm','black':'Black 40 mm'};nets={};paths={};rows=[];coverage=[];checks=[]
 for color in targets:
  modelpath=Path(config['models'][color]['model_path']);paths[color]=str(modelpath)
  for name in ['distance_current','distance_window5']:
   path=modelpath.with_name(name+'.pth');saved=torch.load(path,map_location='cpu',weights_only=True);net=MLP(saved['input_size'],saved['hidden_size']);net.load_state_dict(saved['state_dict']);net.eval();nets[color,name]=(net,saved['scaler'])
 def predict(color,name,x):
  net,s=nets[color,name];x=(np.asarray(x,dtype=np.float32)-s['x_min'])/s['x_scale']
  with torch.inference_mode():return float(net(torch.from_numpy(x).reshape(1,-1)).item()*s['y_scale']+s['y_mean'])
 for seg in manifest['segments']:
  source=list(csv.DictReader((SOURCE/'grasp_segments'/seg['folder']/'predictions.csv').open(encoding='utf-8-sig')));count=0
  for r in source:
   if not all(r[k] for k in ['measured_aperture_px','distance_px','predicted_aperture_px','window_distances_px']):continue
   color=seg['color']
   if r['selected_model_color']!=color:continue
   y=float(r['measured_aperture_px']);p=float(r['predicted_aperture_px']);d=float(r['distance_px'])
   if not np.isfinite([y,p,d]).all():continue
   window=json.loads(r['window_distances_px']);checks.append(abs(p-predict(color,'distance_window5',window)))
   frame=seg['start']+int(r['segment_frame']);phase='rest' if frame<seg['onset'] else 'movement' if frame<seg['stable'] else 'hold'
   rows.append(dict(target=targets[color],trial=seg['folder'],phase=phase,source_frame=frame,actual_aperture_px=y,predicted_current_px=predict(color,'distance_current',[d]),predicted_window5_px=p));count+=1
  coverage.append(dict(trial=seg['folder'],total_frames=len(source),eligible_frames=count,missing_or_unmatched_frames=len(source)-count))
 assert max(checks)<.01,max(checks)
 groups=[*targets.values(),'Overall'];overall=[];per=[];summary=[]
 for target in groups:
  part=[r for r in rows if target=='Overall' or r['target']==target];ids=sorted({r['trial'] for r in part})
  for model,label,key,color in report.MODELS:
   overall.append(dict(target=target,model=model,grasps=len(ids),**report.metrics(part,key)))
   if target!='Overall':
    for trial in ids:
     for phase,_ in report.PHASES:
      subset=[r for r in part if r['trial']==trial and r['phase']==phase]
      if subset:per.append(dict(target=target,trial=trial,model=model,phase=phase,**report.metrics(subset,key)))
 for target in groups:
  for model,_,_,_ in report.MODELS:
   for phase,_ in report.PHASES:
    part=[r for r in per if (target=='Overall' or r['target']==target) and r['model']==model and r['phase']==phase];v=[r['mae'] for r in part]
    summary.append(dict(target=target,model=model,phase=phase,grasps=len(v),frames=sum(r['frames'] for r in part),mean_trial_mae_px=float(np.mean(v)),sd_trial_mae_px=float(np.std(v,ddof=1)) if len(v)>1 else 0.))
 for name,data in [('overall_and_target_metrics',overall),('per_grasp_phase_metrics',per),('phase_summary',summary),('evaluation_frames',rows),('prediction_coverage',coverage)]:report.write_csv(out/(name+'.csv'),data)
 fig,axes=plt.subplots(1,3,figsize=(12,4.5))
 for ax,key,title in zip(axes,['mae','rmse','r2'],['MAE (px)','RMSE (px)','R²']):
  vals=[next(r for r in overall if r['target']=='Overall' and r['model']==m)[key] for m,_,_,_ in report.MODELS];bars=ax.bar(['Current','Window 5'],vals,color=[c for _,_,_,c in report.MODELS]);ax.bar_label(bars,fmt='%.3f' if key=='r2' else '%.2f',padding=3);ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.margins(y=.2)
 report.save_figure(fig,out,'01_overall_model_comparison')
 fig,axes=plt.subplots(1,3,figsize=(15,5))
 for ax,key,title in zip(axes,['mae','rmse','r2'],['MAE (px)','RMSE (px)','R²']):
  for j,(model,label,_,color) in enumerate(report.MODELS):
   vals=[next(r for r in overall if r['target']==g and r['model']==model)[key] for g in groups];ax.bar(np.arange(len(groups))+(j-.5)*.32,vals,width=.30,color=color,label=label)
  ax.set_xticks(np.arange(len(groups)),['Blue\n30 mm','Black\n40 mm','Overall']);ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
 axes[0].legend(fontsize=9);report.save_figure(fig,out,'02_target_and_overall_metrics')
 fig,ax=plt.subplots(figsize=(8,5.5));report.phase_bars(ax,summary,'Overall');ax.set_title('Errors by phase');ax.legend();report.save_figure(fig,out,'03_overall_phase_comparison')
 fig,axes=plt.subplots(1,2,figsize=(10,5),sharey=True)
 for ax,target in zip(axes,targets.values()):report.phase_bars(ax,summary,target);ax.set_title(target)
 axes[0].legend(fontsize=9);report.save_figure(fig,out,'04_phase_comparison_by_target')
 metadata=dict(source=str(SOURCE),models=paths,total_grasps=6,total_frames=sum(r['total_frames'] for r in coverage),eligible_frames=len(rows),window_reconstruction_max_error_px=max(checks),overall='Pooled identical finite recorded-window frames for both models; baseline offline inference from sibling current-distance checkpoint; window uses unchanged realtime predictions.',phase='Initial before reviewed onset; Grasping from onset until reviewed stable hold; Hold from stable hold through segment end. Equal grasp weighting, sample SD ddof=1. No footer.',note='Realtime recording, not offline validation. Blue 3 and Black 3 grasps. Physical black size 40 mm; original model metadata incorrectly says 30 mm. Missing predictions excluded from both metrics and retained in source CSV. No training or smoothing.')
 (out/'sources_and_method.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8');print(out);print(json.dumps([r for r in overall if r['target']=='Overall']));print('Coverage',len(rows),metadata['total_frames'])
if __name__=='__main__':main()
