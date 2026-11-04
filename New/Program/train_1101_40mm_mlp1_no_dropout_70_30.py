"""40mm no-dropout MLP 1f, paired with original signed-px 38/17 run.
Explicit diagnostic: signed midpoint px distance / initial signed value -> aperture in px.
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
class MLP(torch.nn.Module):
 def __init__(self):
  super().__init__()
  self.net=torch.nn.Sequential(torch.nn.Linear(1,32),torch.nn.ReLU(),torch.nn.Identity(),torch.nn.Linear(32,32),torch.nn.ReLU(),torch.nn.Linear(32,1))
 def forward(self,x):return self.net(x).squeeze(-1)


SOURCE=Path(__file__).resolve().parents[2]/'Data/1101_D/40mm'
REFERENCE=SOURCE/'Result/holdout70_30_signed_px_l0_function_mlp5_lstmfull_400ep_20261102_124519_936822'
DISTANCE='hand_object_signed_distance'
TARGET='finger_width'
EPOCHS=400
SEED=20261029
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
 if not rows or any(k not in fields for k in [DISTANCE,TARGET,'frame','t_camera','valid','index_mcp_distance_valid']):raise ValueError(f'{folder}: missing required signed-distance/px fields or reference validity flag')
 def values(k):
  a=[]
  for r in rows:
   try:a.append(float(r[k]))
   except (ValueError,TypeError):a.append(np.nan)
  return np.array(a)
 d,y,frame,t=map(values,[DISTANCE,TARGET,'frame','t_camera'])
 settings=json.loads((folder/'settings.json').read_text(encoding='utf-8-sig'))
 calibration=(settings.get('pixel_calibration') or {}).get('mm_per_px')
 if calibration is None or not np.isfinite(float(calibration)) or float(calibration)<=0:raise ValueError(f'{folder}: missing millimetre calibration; px fallback forbidden')
 for mm,raw in [('hand_object_signed_distance_mm',DISTANCE),('finger_width_mm',TARGET)]:
  if raw not in fields or mm not in fields:raise ValueError(f'{folder}: missing companion field {raw}')
  x,z=values(mm),values(raw);mask=np.isfinite(x)&np.isfinite(z)
  if not np.allclose(x[mask],z[mask]*float(calibration),rtol=1e-5,atol=1e-4):raise ValueError(f'{folder}: inconsistent {mm} calibration')

 if not np.isfinite(t[0]) or not np.isfinite(frame).all() or np.any(frame!=np.floor(frame)) or np.any(np.diff(frame)<=0):raise ValueError(f'{folder}: invalid timestamps/frame IDs')
 valid=np.array([r['valid']=='1' and r.get('index_mcp_distance_valid')=='1' for r in rows])&np.isfinite(d)&np.isfinite(y)&np.isfinite(t)
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
 if not idx:raise ValueError(f'{folder}: no eligible reference5f comparison frames')
 return dict(name=folder.name,folder=folder,frame=frame.astype(int),unix=t,t=t-t[0],d=d,norm=norm,y=y,initial=initial,seqs=[np.array(s) for s in seqs],idx=np.array(idx),windows=np.array(windows)[:,-1:],mm_per_px=float(calibration),csv_hash=digest(p),settings_hash=digest(folder/'settings.json'))
def load(source):
 manifest=json.loads((REFERENCE/'split_manifest.json').read_text(encoding='utf-8-sig'))
 records={(r['split'],r['trial']):r for r in manifest['records']}
 groups={}
 import train_1101_40mm_signed_px_l0_function_mlp_lstm_70_30 as original
 for split,key in [('Train','train'),('Test','test')]:
  group=[]
  for name in manifest[key]:
   folder=source/split/name
   if not folder.is_dir():raise ValueError(f'Missing paired trial: {folder}')
   g=read_trial(folder)
   if g['csv_hash']!=records[(split,name)]['csv_sha256']:raise ValueError(f'{name}: data changed since baseline; paired run stopped')
   baseline=original.load_trial(folder)
   assert np.array_equal(g['idx'],baseline['common'])
   assert np.array_equal(g['y'],baseline['y'],equal_nan=True)
   group.append(g)
  groups[split]=group
 assert len(groups['Train'])==38 and len(groups['Test'])==17
 keys=lambda trials:{(int(f),float(t)) for g in trials for f,t in zip(g['frame'],g['unix']) if np.isfinite(t)}
 assert not keys(groups['Train'])&keys(groups['Test'])
 return groups

def make_model(device='cpu'):
 return MLP().to(device)

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
   full=np.full(len(g['y']),np.nan)
   valid=np.concatenate(g['seqs'])
   with torch.inference_mode():full[valid]=model(torch.tensor(g['norm'][valid,None],dtype=torch.float32,device=device)).cpu().numpy()*scale['y_std']+scale['y_mean']
   np.testing.assert_allclose(full[idx],p,atol=1e-4,rtol=1e-5)
   initial.append(dict(split=split,trial=g['name'],source_frame=int(g['frame'][valid[0]]),actual_px=float(g['y'][valid[0]]),predicted_px=float(full[valid[0]]),error_px=float(full[valid[0]]-g['y'][valid[0]])))
   write_csv(output/split/'predictions'/f'{g["name"]}.csv',[dict(source_frame=int(g['frame'][i]),time_s=float(g['t'][i]),signed_distance_px=float(g['d'][i]),distance_over_initial=float(g['norm'][i]),initial_signed_distance_px=g['initial'],actual_aperture_px=float(g['y'][i]),mlp_1f_px=float(full[i]),valid_prediction=int(np.isfinite(full[i])),common_comparison_frame=int(i in idx)) for i in range(len(full))])
   fig,axs=plt.subplots(1,2,figsize=(12,4.3))
   for k,seq in enumerate(g['seqs']):
    for ax,x in zip(axs,[g['t'],g['norm']]):
     ax.plot(x[seq],g['y'][seq],color='black',lw=1.4,label='Measured' if k==0 else None)
     ax.plot(x[seq],full[seq],color='#e58b26',lw=1.3,label='MLP 1f' if k==0 else None)
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
 ax.set(xlabel='Epoch',ylabel='Train MAE (px)',title='MLP 1f');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(output/'training_curve.png',dpi=150);plt.close(fig)
 info=dict(architecture='MLP 1f: Linear(1,32), ReLU, Identity (no dropout), Linear(32,32), ReLU, Linear(32,1)',input_column=DISTANCE,input_processing='Original signed midpoint distance divided by first recorded value; negative distances preserved, no clipping or absolute value; no secondary input scaling',output_column=TARGET,units=dict(input='dimensionless signed ratio',output='px',metrics='px; R2 dimensionless',time='seconds'),numeric_scaler=scale,epochs=0 if simulated else EPOCHS,simulated_untrained=simulated,dropout=0.0,reference_run=str(REFERENCE),seed=SEED,training='Current implementation: shuffle trial order; one Adam step per complete trial; equal trial mean absolute loss; lr=0.001, weight_decay=0.0001',selection='Fixed 400 epochs; last epoch saved; no Test used for fitting, tuning, or epoch selection',window='Only current signed-distance ratio is input, shape(N,1), no history or padding. For a controlled window-length comparison, fitting and main metrics retain the original5f eligible target frames. Separate full predictions start at the first valid frame.',split='Exact38/17 names, order, main target frames and target scaler from signed-px baseline; input window reduced5to1, no dropout',diagnostic_scope='Same data, split, main target frames, scaler, seed and optimizer settings; current-frame input only, no dropout. First layer is1to32 rather than5to32, so initialization cannot be identical across input shapes. 400 fixed epochs; final weights saved, no Test selection.')
 write_json(output/'model_information.json',info)
 (output/'models').mkdir(exist_ok=True)
 torch.save(dict(info,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}),output/'models/mlp_1f.pth')

def compare_baseline(output,groups,model,scale,device):
 expected={(split,g['name']):g['csv_hash'] for split,group in groups.items() for g in group}
 comparison_run=None
 for candidate in sorted((SOURCE/'Result').glob('mlp5_no_dropout_signed_px_l0_70_30_400ep_*'),reverse=True):
  manifest=candidate/'split_manifest.json';info=candidate/'model_information.json'
  if not manifest.is_file() or not info.is_file():continue
  meta=json.loads(info.read_text())
  if meta.get('simulated_untrained') or meta.get('dropout')!=0:continue
  records=json.loads(manifest.read_text())['records']
  if {(r['split'],r['trial']):r['csv_sha256'] for r in records}==expected:
   if all((candidate/split/'predictions'/f"{g['name']}.csv").is_file() for split,group in groups.items() for g in group):comparison_run=candidate;break
 if comparison_run is None:
  print('No completed matching no-dropout5f run found; skipped optional comparison.',flush=True);return
 rows=[]
 for split,trials in groups.items():
  y_all=[];old_all=[];new_all=[]
  for g in trials:
   with (comparison_run/split/'predictions'/f"{g['name']}.csv").open(encoding='utf-8-sig') as f:old={int(r['source_frame']):r for r in csv.DictReader(f)}
   idx=g['idx'];p=predict(model,g,scale,device)
   meta=json.loads((g['folder']/'segment.json').read_text(encoding='utf-8-sig'));onset=meta['onset_row']-meta['start_row'];mask=idx<onset
   old_p=np.array([float(old[int(g['frame'][i])]['mlp_5f_px']) for i in idx])
   y_all.extend(g['y'][idx[mask]]);old_all.extend(old_p[mask]);new_all.extend(p[mask])
  y=np.array(y_all)
  for label,values in [('MLP5f no dropout',old_all),('MLP1f no dropout',new_all)]:
   values=np.array(values);rows.append(dict(split=split,model=label,trials=len(trials),initial_rest_frames=len(y),actual_mean_px=float(y.mean()),predicted_mean_px=float(values.mean()),**metrics(y,values)))
 write_csv(output/'initial_rest_1f_vs_5f.csv',rows)
 write_json(output/'comparison_reference.json',dict(reference=str(comparison_run),selection='Latest completed matching no-dropout5f run by filename timestamp, never selected by Test metrics',frames='Identical pre-onset frames eligible for5f comparison'))
 print('Saved optional1f-vs5f initial-rest comparison:',comparison_run,flush=True)

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,default=SOURCE);parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');parser.add_argument('--check-data',action='store_true');parser.add_argument('--smoke-export',action='store_true');args=parser.parse_args()
 if args.smoke_export and not args.check_data:parser.error('--smoke-export requires --check-data')
 source=args.source.resolve()
 try:groups=load(source)
 except (ValueError,OSError,KeyError) as exc:parser.error(str(exc))
 train,test=groups['Train'],groups['Test'];print(f'{len(train)} Train / {len(test)} Test | same signed-px baseline split (38/17)',flush=True)
 print('MLP 1f NO DROPOUT | signed midpoint distance / initial distance -> finger_width (px) | 400 epochs, final epoch',flush=True)
 torch.set_num_threads(min(4,torch.get_num_threads()));random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(SEED)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 y=np.concatenate([g['y'][g['idx']] for g in train]);scale=dict(y_mean=float(y.mean()),y_std=max(float(y.std()),1e-6),x_mean=0.0,x_std=1.0)
 if args.check_data:
  model=make_model();before={k:v.clone() for k,v in model.state_dict().items()}
  assert all(g['norm'][0]==1 and np.allclose(g['norm']*g['initial'],g['d'],equal_nan=True) for group in groups.values() for g in group)
  assert all(g['windows'].shape[1]==1 for group in groups.values() for g in group)
  for g in test:assert np.isfinite(predict(model,g,scale,'cpu')).all()
  if args.smoke_export:
   with tempfile.TemporaryDirectory(prefix='SIGNED_PX_MLP_UNTRAINED_',dir=Path(__file__).parent) as temporary:
    out=Path(temporary);export(out,{'Train':train[:1],'Test':test[:1]},model,scale,[],'cpu',True)
    checkpoint=torch.load(out/'models/mlp_1f.pth',map_location='cpu',weights_only=True);restored=MLP();restored.load_state_dict(checkpoint['state_dict'])
    for k,v in before.items():assert torch.equal(restored.state_dict()[k],v)
  for k,v in before.items():assert torch.equal(model.state_dict()[k],v)
  print('PASS: signed fields, initial-only normalization, current-frame-only input, identical5f comparison targets, disjoint trials/frames, train-only target scaler, model/export roundtrip. No training or persistent Result outputs.',flush=True)
 else:
  if args.device=='cuda' and not torch.cuda.is_available():parser.error('CUDA unavailable')
  device=torch.device('cuda' if args.device=='cuda' or (args.device=='auto' and torch.cuda.is_available()) else 'cpu')
  output=source/'Result'/f'mlp1_no_dropout_signed_px_l0_70_30_400ep_{datetime.now():%Y%m%d_%H%M%S_%f}';output.mkdir(parents=True)
  write_json(output/'split_manifest.json',dict(reference_run=str(REFERENCE),seed=SEED,strategy='Reference40mm signed-px38/17 reused; no cross-validation',records=[dict(split=s,trial=g['name'],source=str(g['folder']),csv_sha256=g['csv_hash']) for s,group in groups.items() for g in group]))
  model=make_model(device);optimizer=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4);rng=random.Random(SEED);history=[]
  print(f'Device: {device}; training only MLP 1f...',flush=True)
  for epoch in range(1,EPOCHS+1):
   model.train();order=list(range(len(train)));rng.shuffle(order);total=0.
   for number in order:
    g=train[number];optimizer.zero_grad(set_to_none=True);pred=model(torch.tensor(g['windows'],dtype=torch.float32,device=device));target=torch.tensor((g['y'][g['idx']]-scale['y_mean'])/scale['y_std'],dtype=torch.float32,device=device)
    loss=(pred-target).abs().mean();loss.backward();optimizer.step();total+=float(loss.detach())*scale['y_std']
   history.append(dict(epoch=epoch,train_mae_px=total/len(train)))
   if epoch==1 or epoch%25==0:print(f'MLP 1f: epoch {epoch}/{EPOCHS}, Train MAE {history[-1]["train_mae_px"]:.4f} px',flush=True)
  export(output,groups,model,scale,history,device)
  compare_baseline(output,groups,model,scale,device)
  print(f'Saved final epoch and offline results: {output}',flush=True)
 for group in groups.values():
  for g in group:
   assert digest(g['folder']/'camera.csv')==g['csv_hash'] and digest(g['folder']/'settings.json')==g['settings_hash']

if __name__=='__main__':main()
