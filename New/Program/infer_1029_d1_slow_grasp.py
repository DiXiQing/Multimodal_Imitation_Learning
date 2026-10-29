"""Offline inference only: saved normalized Function / MLP 5f / LSTM Full.
Default: held-out black grasp_052, actual movement 4.158 s. Never trains.
--check-only runs inference/validation without creating result files.
"""
from __future__ import annotations
import argparse,csv,json,hashlib
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from torch import nn
from scipy.interpolate import BSpline
BASE=Path(__file__).resolve().parents[2]
TRIAL=BASE/'Data/1029_D1/40mm_black/backup/grasp_052'
RUN=BASE/'Data/1029_D1/40mm_black/Result/l0_normalized_function_mlp5_lstmfull_400ep_20261029_164218_197057'
LABELS=['Function','MLP 5f','LSTM Full'];COLORS=['#3485b8','#e59a32','#519e58']
class MLP(nn.Module):
 def __init__(self,hidden=32):
  super().__init__();self.net=nn.Sequential(nn.Linear(5,hidden),nn.ReLU(),nn.Dropout(.1),nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,1))
 def forward(self,x):return self.net(x).squeeze(-1)
class FullLSTM(nn.Module):
 def __init__(self,hidden=32):
  super().__init__();self.lstm=nn.LSTM(1,hidden,batch_first=True);self.head=nn.Sequential(nn.Dropout(.1),nn.Linear(hidden,1))
 def forward(self,x,state=None):
  sequence,state=self.lstm(x,state);return self.head(sequence).squeeze(-1),state

def read_csv(file):
 with file.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def digest(file):return hashlib.sha256(file.read_bytes()).hexdigest()
def score(y,p):
 err=p-y;den=float(np.sum((y-y.mean())**2))
 return dict(mae_mm=float(abs(err).mean()),rmse_mm=float(np.sqrt(np.mean(err**2))),r2=float(1-np.sum(err**2)/den) if den>1e-12 else None)
def save_csv(file,rows):
 with file.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--trial',type=Path,default=TRIAL);parser.add_argument('--run',type=Path,default=RUN)
 parser.add_argument('--output',type=Path,help='New output folder only; existing folders are never overwritten')
 parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');parser.add_argument('--check-only',action='store_true')
 args=parser.parse_args();trial=args.trial.resolve();run=args.run.resolve()
 paths=[trial/'camera.csv',trial/'settings.json',trial/'segment.json',run/'models/function.json',run/'models/mlp_5f.pth',run/'models/lstm_full.pth']
 for file in paths:
  if not file.is_file():parser.error(f'Missing file: {file}')
 hashes={str(file):digest(file) for file in paths}
 rows=read_csv(paths[0]);settings=json.loads(paths[1].read_text(encoding='utf-8-sig'));meta=json.loads(paths[2].read_text(encoding='utf-8-sig'))
 if not rows:parser.error('Empty camera.csv')
 for field in ['frame','t_camera','index_mcp_object_distance_mm','finger_width_mm','valid','index_mcp_distance_valid']:
  if field not in rows[0]:parser.error(f'Missing required column {field}; no px fallback')
 try:scale=float((settings.get('pixel_calibration') or {})['mm_per_px'])
 except (KeyError,TypeError,ValueError):parser.error('Valid mm calibration is required; no px fallback')
 if not np.isfinite(scale) or scale<=0:parser.error('Invalid mm calibration')
 def values(key):
  def number(r):
   try:return float(r[key])
   except (ValueError,TypeError):return np.nan
  return np.array([number(r) for r in rows])
 d=values('index_mcp_object_distance_mm');y=values('finger_width_mm');t=values('t_camera');frame=values('frame')
 valid=np.array([r['valid']=='1' and r['index_mcp_distance_valid']=='1' for r in rows])
 if not valid[0] or not np.isfinite(d[0]) or d[0]<=0 or not np.isfinite(t[0]):parser.error('First-frame distance/time invalid; cannot define initial distance')
 if not np.all(np.isfinite(frame)) or len(set(frame))!=len(frame) or np.any(frame!=np.floor(frame)):parser.error('Invalid/duplicate source frame IDs')
 for mm,px in [('index_mcp_object_distance_mm','index_mcp_object_distance'),('finger_width_mm','finger_width')]:
  if px in rows[0]:
   a=values(mm);b=values(px);mask=np.isfinite(a)&np.isfinite(b)
   if not np.allclose(a[mask],b[mask]*scale,rtol=1e-5,atol=1e-4):parser.error(f'{mm} inconsistent with settings calibration')
 if args.device=='cuda' and not torch.cuda.is_available():parser.error('CUDA unavailable')
 device=torch.device('cuda' if args.device=='cuda' or (args.device=='auto' and torch.cuda.is_available()) else 'cpu');torch.set_num_threads(min(4,torch.get_num_threads()))
 function=json.loads(paths[3].read_text(encoding='utf-8-sig'));mlp_ck=torch.load(paths[4],map_location='cpu',weights_only=True);lstm_ck=torch.load(paths[5],map_location='cpu',weights_only=True)
 for ck in [function,mlp_ck,lstm_ck]:
  if ck.get('simulated_untrained') or 'first recorded frame' not in ck.get('physical_normalization',''):parser.error('Expected trained initial-distance-normalized checkpoints')
  if ck.get('input_column')!='index_mcp_object_distance_mm' or ck.get('target_column')!='finger_width_mm':parser.error('Checkpoint input/output columns do not match')
 for ck in [mlp_ck,lstm_ck]:
  if ck.get('epochs')!=400:parser.error('Expected final 400-epoch checkpoint')
 scaler=mlp_ck['numeric_scaler']
 if any(ck['numeric_scaler']!=scaler for ck in [function,lstm_ck]):parser.error('Checkpoint numerical scalers differ')
 max_gap=float(mlp_ck.get('max_frame_gap_s',.10))
 if not np.isfinite(max_gap) or max_gap<=0:parser.error('Invalid checkpoint frame gap')
 # Verify independence from BOTH training and existing test experiments.
 sample_keys={(int(f),float(stamp)) for f,stamp in zip(frame,t) if np.isfinite(stamp)};overlaps=[];dataset=run.parent.parent
 for split in ['Train','Test']:
  for csvfile in (dataset/split).glob('grasp_*/camera.csv'):
   other=read_csv(csvfile);other_keys={(int(r['frame']),float(r['t_camera'])) for r in other}
   if sample_keys & other_keys:overlaps.append(f'{split}/{csvfile.parent.name}')
 if overlaps:parser.error('Selected trial overlaps existing Train/Test: '+', '.join(overlaps))
 # Object settings must match the model's dataset (calibration scales may differ by recording).
 examples=list((dataset/'Train').glob('grasp_*/settings.json'))
 if examples:
  trained_settings=json.loads(examples[0].read_text(encoding='utf-8-sig'))
  if any(settings.get(k)!=trained_settings.get(k) for k in ['object_color','size_cm','width','height']):parser.error('Trial object class/resolution differs from model dataset')
 normalized=d/d[0];assert normalized[0]==1.
 usable=valid&np.isfinite(d)&np.isfinite(y)&np.isfinite(t);seqs=[];current=[]
 for i in range(len(rows)):
  if not usable[i]:
   if current:seqs.append(current);current=[]
   continue
  if current and (frame[i]-frame[current[-1]]!=1 or not 0<t[i]-t[current[-1]]<=max_gap):seqs.append(current);current=[]
  current.append(i)
 if current:seqs.append(current)
 if not seqs:parser.error('No valid continuous sequences')
 models={'MLP 5f':MLP(mlp_ck.get('hidden_size',32)).to(device),'LSTM Full':FullLSTM(lstm_ck.get('hidden_size',32)).to(device)}
 for label,ck in [('MLP 5f',mlp_ck),('LSTM Full',lstm_ck)]:models[label].load_state_dict(ck['state_dict']);models[label].eval()
 def tx(value):return torch.tensor((np.asarray(value)-scaler['x_mean'])/scaler['x_std'],dtype=torch.float32,device=device)
 def undo(value):return np.asarray(value)*scaler['y_std']+scaler['y_mean']
 spec=function['definition'];spline=BSpline(spec['knots'],spec['coefficients'],spec['degree'],extrapolate=True)
 pred={label:np.full(len(rows),np.nan) for label in LABELS};common=np.zeros(len(rows),bool)
 with torch.inference_mode():
  for seq in seqs:
   pred['Function'][seq]=undo(spline((normalized[seq]-scaler['x_mean'])/scaler['x_std']))
   state=None
   for i in seq:
    estimate,state=models['LSTM Full'](tx([normalized[i]]).reshape(1,1,1),state);pred['LSTM Full'][i]=float(undo(estimate[0,0].cpu().item()))
   for k in range(4,len(seq)):
    i=seq[k];estimate=models['MLP 5f'](tx(normalized[seq[k-4:k+1]]).reshape(1,5));pred['MLP 5f'][i]=float(undo(estimate[0].cpu().item()));common[i]=True
 if not common.any():parser.error('No common valid five-frame targets')
 metrics=[dict(model=label,frames=int(common.sum()),**score(y[common],pred[label][common])) for label in LABELS]
 full=np.isfinite(pred['LSTM Full'])&np.isfinite(y)
 full_metric=dict(model='LSTM Full',frames=int(full.sum()),**score(y[full],pred['LSTM Full'][full]))
 onset=meta.get('onset_row');closure=meta.get('closure_confirmed_row');start=meta.get('start_row',0);motion=None
 if onset is not None and closure is not None and 0<=onset-start<=closure-start<len(rows):motion=float(t[closure-start]-t[onset-start])
 report=dict(trial=str(trial),model_run=str(run),initial_distance_mm=float(d[0]),normalization='current distance / first recorded distance',total_duration_s=float(t[-1]-t[0]),movement_duration_s=motion,movement_definition='Annotated onset to five-frame closure confirmation, excluding pre-roll/post-hold',source_frames=len(rows),common_frames=int(common.sum()),continuous_sequences=len(seqs),gap_or_missing_resets=max(0,len(seqs)-1),max_gap_s=max_gap,review_flags=meta.get('review_flags',[]),independent_of_existing_train_and_test=True,common_frame_metrics=metrics,lstm_full_metrics=full_metric,training_performed=False,original_file_sha256=hashes)
 print(json.dumps({k:v for k,v in report.items() if k!='original_file_sha256'},indent=2),flush=True)
 if not args.check_only:
  output=args.output.resolve() if args.output else run.parent/f'offline_slow_inference_{trial.name}_{datetime.now():%Y%m%d_%H%M%S_%f}'
  output.mkdir(parents=True,exist_ok=False)
  output_rows=[]
  for i,r in enumerate(rows):output_rows.append(dict(r,offline_elapsed_s=float(t[i]-t[0]),distance_over_initial=float(normalized[i]),common_valid=int(common[i]),predicted_function_mm=float(pred['Function'][i]),predicted_mlp_5f_mm=float(pred['MLP 5f'][i]),predicted_lstm_full_mm=float(pred['LSTM Full'][i])))
  save_csv(output/'predictions.csv',output_rows);save_csv(output/'metrics_common.csv',metrics);save_csv(output/'lstm_full_predictions.csv',[output_rows[i] for i in np.flatnonzero(full)]);save_csv(output/'lstm_full_metrics.csv',[full_metric])
  import os
  os.environ.setdefault('MPLCONFIGDIR',str(BASE/'tool/.mplconfig'))
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig,axes=plt.subplots(1,2,figsize=(13,4.5))
  for ax,x,xlabel in zip(axes,[t-t[0],normalized],['Actual elapsed time (s)','Normalized distance d/d0']):
   for j,seq in enumerate(seqs):
    ax.plot(x[seq],y[seq],color='black',lw=1.8,label='Measured' if j==0 else None)
    idx=[i for i in seq if common[i]]
    for label,color in zip(LABELS,COLORS):ax.plot(x[idx],pred[label][idx],color=color,label=label if j==0 else None)
   ax.set_xlabel(xlabel);ax.set_ylabel('Finger aperture (mm)');ax.grid(alpha=.2);ax.legend()
  fig.suptitle(f'{trial.name} | held-out slow grasp | movement {motion:.3f} s' if motion is not None else trial.name);fig.tight_layout();fig.savefig(output/'prediction_comparison.png',dpi=180);plt.close(fig)
  (output/'inference_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print('Saved NEW inference output:',output,flush=True)
 assert all(digest(Path(file))==h for file,h in hashes.items()),'Original input/model files changed'
 print('VERIFIED: original files unchanged; no training or optimizer steps.',flush=True)
if __name__=='__main__':main()
