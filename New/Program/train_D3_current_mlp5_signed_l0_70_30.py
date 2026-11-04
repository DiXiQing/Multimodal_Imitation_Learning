"""Current MLP 5f on D3's original signed midpoint distance; fixed 63/27 split.
Explicit diagnostic: signed px distance / its initial value -> aperture in px.
--check-data --smoke-export checks without fitting or training; temporary exports only.
"""
from pathlib import Path
from datetime import datetime
import argparse,csv,json,random,hashlib,tempfile,os
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).parent/'.mplconfig'))
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from train_1031_d_mixed_l0_function_mlp_lstm import MLP

SOURCE=Path(__file__).resolve().parents[2]/'Data/D3'
DISTANCE='hand_object_signed_distance'
TARGET='finger_width'
EPOCHS=400
SEED=20261030
GAP=.1

def write_json(p,value):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def write_csv(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 if not rows:return
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read_trial(folder):
 p=folder/'camera.csv'
 with p.open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
 if not rows or any(k not in fields for k in [DISTANCE,TARGET,'frame','t_camera','valid']):raise ValueError(f'{folder}: missing required original signed-distance/px fields')
 def values(k):
  a=[]
  for r in rows:
   try:a.append(float(r[k]))
   except (ValueError,TypeError):a.append(np.nan)
  return np.array(a)
 d,y,frame,t=map(values,[DISTANCE,TARGET,'frame','t_camera'])
 if not np.isfinite(t[0]) or not np.isfinite(frame).all() or np.any(frame!=np.floor(frame)) or np.any(np.diff(frame)<=0):raise ValueError(f'{folder}: invalid timestamps/frame IDs')
 valid=np.array([r['valid']=='1' for r in rows])&np.isfinite(d)&np.isfinite(y)&np.isfinite(t)
 if not valid[0] or d[0]<=0:raise ValueError(f'{folder}: invalid/nonpositive signed initial distance; no later-frame fallback')
 initial=float(d[0]);norm=d/initial;seqs=[];current=[]
 for i in range(len(rows)):
  if not valid[i]:
   if current:seqs.append(current);current=[]
   continue
  if current and (frame[i]-frame[current[-1]]!=1 or not 0<t[i]-t[current[-1]]<=GAP):seqs.append(current);current=[]
  current.append(i)
 if current:seqs.append(current)
 idx=[];windows=[]
 for seq in seqs:
  for k in range(4,len(seq)):
   selected=seq[k-4:k+1];idx.append(seq[k]);windows.append(norm[selected])
 if not idx:raise ValueError(f'{folder}: no valid consecutive five-frame windows')
 return dict(name=folder.name,folder=folder,frame=frame.astype(int),unix=t,t=t-t[0],d=d,norm=norm,y=y,initial=initial,seqs=[np.array(s) for s in seqs],idx=np.array(idx),windows=np.array(windows),csv_hash=digest(p),settings_hash=digest(folder/'settings.json'))
def load(source):
 groups={}
 for output,sub in [('Train','Train'),('Test','Validation')]:
  parent=source/sub
  if not parent.is_dir():raise ValueError(f'Missing existing D3 split directory: {parent}')
  folders=sorted(p for p in parent.iterdir() if p.is_dir() and (p/'camera.csv').is_file() and (p/'segment.json').is_file())
  groups[output]=[read_trial(p) for p in folders]
 train,test=groups['Train'],groups['Test']
 if not train or not test:raise ValueError('Empty training or test partition')
 if {g['name'] for g in train}&{g['name'] for g in test}:raise ValueError('Overlapping trial names')
 if {g['csv_hash'] for g in train}&{g['csv_hash'] for g in test}:raise ValueError('Identical Train/Test CSVs')
 keys=lambda trials:{(int(f),float(t)) for g in trials for f,t in zip(g['frame'],g['unix']) if np.isfinite(t)}
 if keys(train)&keys(test):raise ValueError('Recorded-frame leakage')
 return groups
def metrics(y,p):
 e=p-y;den=float(np.sum((y-y.mean())**2))
 return dict(mae_px=float(abs(e).mean()),rmse_px=float(np.sqrt(np.mean(e**2))),r2=float(1-np.sum(e**2)/den) if den>1e-12 else None,bias_px=float(e.mean()))
def predict(model,g,scale,device):
 model.eval()
 with torch.inference_mode():return model(torch.tensor(g['windows'],dtype=torch.float32,device=device)).cpu().numpy()*scale['y_std']+scale['y_mean']
def export(output,groups,model,scale,history,device,simulated=False):
 per=[];overall=[];initial=[]
 for split,trials in groups.items():
  truth=[];predictions=[]
  for g in trials:
   p=predict(model,g,scale,device);idx=g['idx'];truth.extend(g['y'][idx]);predictions.extend(p)
   per.append(dict(split=split,trial=g['name'],frames=len(idx),**metrics(g['y'][idx],p)))
   initial.append(dict(split=split,trial=g['name'],source_frame=int(g['frame'][idx[0]]),actual_px=float(g['y'][idx[0]]),predicted_px=float(p[0]),error_px=float(p[0]-g['y'][idx[0]])))
   full=np.full(len(g['y']),np.nan);full[idx]=p
   write_csv(output/split/'predictions'/f'{g["name"]}.csv',[dict(source_frame=int(g['frame'][i]),time_s=float(g['t'][i]),signed_distance_px=float(g['d'][i]),distance_over_initial=float(g['norm'][i]),initial_signed_distance_px=g['initial'],actual_aperture_px=float(g['y'][i]),mlp_5f_px=float(full[i]),valid_prediction=int(np.isfinite(full[i]))) for i in range(len(full))])
   fig,axs=plt.subplots(1,2,figsize=(12,4.3))
   for k,seq in enumerate(g['seqs']):
    for ax,x in zip(axs,[g['t'],g['norm']]):
     ax.plot(x[seq],g['y'][seq],color='black',lw=1.4,label='Measured' if k==0 else None)
     ax.plot(x[seq],full[seq],color='#e58b26',lw=1.3,label='MLP 5f' if k==0 else None)
   for ax,xlabel in zip(axs,['Time (s)',r'Normalized signed distance $s_t/s_0$']):
    ax.set_xlabel(xlabel);ax.set_ylabel('Finger aperture (px)');ax.legend();ax.grid(alpha=.2)
   fig.suptitle(('UNTRAINED CHECK ONLY | ' if simulated else '')+f'{split}: {g["name"]}');fig.tight_layout();fig.savefig(output/split/'predictions'/f'{g["name"]}.png',dpi=150);plt.close(fig)
  overall.append(dict(split=split,trials=len(trials),frames=len(truth),**metrics(np.array(truth),np.array(predictions))))
 write_csv(output/'per_trial_metrics.csv',per);write_csv(output/'overall_metrics.csv',overall);write_csv(output/'initial_prediction_errors.csv',initial);write_csv(output/'training_history.csv',history)
 summary=[]
 for split in groups:
  r=[x for x in per if x['split']==split];v=np.array([x['mae_px'] for x in r])
  summary.append(dict(split=split,trials=len(r),trial_mae_mean_px=float(v.mean()),trial_mae_sd_px=float(v.std(ddof=1)) if len(v)>1 else None))
 write_csv(output/'trial_mean_sd.csv',summary)
 fig,ax=plt.subplots(figsize=(7,4))
 if history:ax.plot([r['epoch'] for r in history],[r['train_mae_px'] for r in history])
 else:ax.text(.5,.5,'UNTRAINED: export check only',ha='center',transform=ax.transAxes)
 ax.set(xlabel='Epoch',ylabel='Train MAE (px)',title='MLP 5f');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(output/'training_curve.png',dpi=150);plt.close(fig)
 info=dict(architecture='MLP 5f: Linear(5,32), ReLU, Dropout(0.1), Linear(32,32), ReLU, Linear(32,1)',input_column=DISTANCE,input_processing='Original signed midpoint distance divided by first recorded value; negative distances preserved, no clipping or absolute value; no secondary input scaling',output_column=TARGET,units=dict(input='dimensionless signed ratio',output='px',metrics='px; R2 dimensionless',time='seconds'),numeric_scaler=scale,epochs=0 if simulated else EPOCHS,simulated_untrained=simulated,seed=SEED,training='Current implementation: shuffle trial order; one Adam step per complete trial; equal trial mean absolute loss; lr=0.001, weight_decay=0.0001',selection='Fixed 400 epochs; last epoch saved; no Test used for fitting, tuning, or epoch selection',window='Five consecutive real inputs oldest-to-current; no initial padding; discard windows across missing/invalid rows or frame/time gaps >0.1s',split='Existing D3 Train / Validation reused as Train / Test; whole trials; no five-fold CV',diagnostic_scope='Legacy signed-distance experiment requested explicitly. No landmark-5 or mm fallback. D3 differs from current recordings in multiple ways; this comparison alone cannot isolate a causal effect of changing the landmark.')
 write_json(output/'model_information.json',info)
 (output/'models').mkdir(exist_ok=True)
 torch.save(dict(info,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}),output/'models/mlp_5f.pth')

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,default=SOURCE);parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');parser.add_argument('--check-data',action='store_true');parser.add_argument('--smoke-export',action='store_true');args=parser.parse_args()
 if args.smoke_export and not args.check_data:parser.error('--smoke-export requires --check-data')
 source=args.source.resolve()
 try:groups=load(source)
 except (ValueError,OSError,KeyError) as exc:parser.error(str(exc))
 train,test=groups['Train'],groups['Test'];print(f'{len(train)} Train / {len(test)} Test | existing whole-trial 7/3 split',flush=True)
 print('MLP 5f only | signed midpoint distance / initial distance -> finger_width (px) | 400 epochs, final epoch',flush=True)
 torch.set_num_threads(min(4,torch.get_num_threads()));random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(SEED)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 y=np.concatenate([g['y'][g['idx']] for g in train]);scale=dict(y_mean=float(y.mean()),y_std=max(float(y.std()),1e-6),x_mean=0.0,x_std=1.0)
 if args.check_data:
  model=MLP();before={k:v.clone() for k,v in model.state_dict().items()}
  assert all(g['norm'][0]==1 and np.allclose(g['norm']*g['initial'],g['d'],equal_nan=True) for group in groups.values() for g in group)
  assert all(g['windows'].shape[1]==5 for group in groups.values() for g in group)
  for g in test:assert np.isfinite(predict(model,g,scale,'cpu')).all()
  if args.smoke_export:
   with tempfile.TemporaryDirectory(prefix='D3_MLP_UNTRAINED_',dir=Path(__file__).parent) as temporary:
    out=Path(temporary);export(out,{'Train':train[:1],'Test':test[:1]},model,scale,[],'cpu',True)
    checkpoint=torch.load(out/'models/mlp_5f.pth',map_location='cpu',weights_only=True);restored=MLP();restored.load_state_dict(checkpoint['state_dict'])
    for k,v in before.items():assert torch.equal(restored.state_dict()[k],v)
  for k,v in before.items():assert torch.equal(model.state_dict()[k],v)
  print('PASS: signed fields, initial-only normalization, real 5-frame causal windows, disjoint trials/frames, train-only target scaler, model/export roundtrip. No training or persistent Result outputs.',flush=True)
 else:
  if args.device=='cuda' and not torch.cuda.is_available():parser.error('CUDA unavailable')
  device=torch.device('cuda' if args.device=='cuda' or (args.device=='auto' and torch.cuda.is_available()) else 'cpu')
  output=source/'Result'/f'current_mlp5_signed_l0_400ep_{datetime.now():%Y%m%d_%H%M%S_%f}';output.mkdir(parents=True)
  write_json(output/'split_manifest.json',dict(seed=SEED,strategy='Existing D3 Train / Validation, reused as Train / Test',records=[dict(split=s,trial=g['name'],source=str(g['folder']),csv_sha256=g['csv_hash']) for s,group in groups.items() for g in group]))
  model=MLP().to(device);optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4);rng=random.Random(SEED);history=[]
  print(f'Device: {device}; training only MLP 5f...',flush=True)
  for epoch in range(1,EPOCHS+1):
   model.train();order=list(range(len(train)));rng.shuffle(order);total=0.
   for number in order:
    g=train[number];optimizer.zero_grad(set_to_none=True);pred=model(torch.tensor(g['windows'],dtype=torch.float32,device=device));target=torch.tensor((g['y'][g['idx']]-scale['y_mean'])/scale['y_std'],dtype=torch.float32,device=device)
    loss=(pred-target).abs().mean();loss.backward();optimizer.step();total+=float(loss.detach())*scale['y_std']
   history.append(dict(epoch=epoch,train_mae_px=total/len(train)))
   if epoch==1 or epoch%25==0:print(f'MLP 5f: epoch {epoch}/{EPOCHS}, Train MAE {history[-1]["train_mae_px"]:.4f} px',flush=True)
  export(output,groups,model,scale,history,device)
  print(f'Saved final epoch and offline results: {output}',flush=True)
 for group in groups.values():
  for g in group:
   assert digest(g['folder']/'camera.csv')==g['csv_hash'] and digest(g['folder']/'settings.json')==g['settings_hash']

if __name__=='__main__':main()
