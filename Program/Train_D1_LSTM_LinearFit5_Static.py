"""Train a D1-only LinearFit5 LSTM with randomized rest prefixes and rest-only sequences.
Uses one existing held-out fold; copies its unchanged Function/MLP into a new live bundle.
"""
import argparse,csv,json,random,shutil,time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from torch import nn
import Train_D1_Joint_Three_Models_LinearFit5_CV5 as base
ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'Data/1101_D_PreAction15/Result/Joint_Three_Models_LinearFit5_CurrentV_CV5_20261123_153838'

def baseline_aperture(g):
 return float(np.median(g['y'][:5]))

def prepend_rest(g,seconds,scaling):
 """Synthetic frames only in memory; original L0 and all original targets preserved."""
 first=g['seqs'][0]
 if seconds<=0 or int(first[0])!=4:return g,0
 dt=float(np.median(np.diff(g['t'][:5])));n=max(0,round(seconds/dt))
 if not n:return g,0
 times=np.r_[-dt*np.arange(n,0,-1),g['t']]
 distance=np.r_[np.full(n,g['initial_distance_mm']),g['d_mm']]
 runs=[np.arange(int(first[-1])+n+1)]
 runs.extend(np.arange(int(run[0])-4,int(run[-1])+1)+n for run in g['seqs'][1:])
 velocity=base.estimate_velocity(distance,times,runs)
 trial=dict(g,t=times,d=distance/g['initial_distance_mm'],d_mm=distance,
            y=np.r_[np.full(n,baseline_aperture(g)),g['y']],
            seqs=[run[4:] for run in runs],
            velocity_mm_s=velocity,velocity_scaled=(velocity-scaling['mean_mm_s'])/scaling['std_mm_s'])
 return trial,n

def rest_only(g,seconds,scaling):
 dt=float(np.median(np.diff(g['t'][:5])));n=max(5,round(seconds/dt))
 return dict(g,t=np.arange(n)*dt,d=np.ones(n),y=np.full(n,baseline_aperture(g)),
             velocity_mm_s=np.zeros(n),velocity_scaled=np.full(n,-scaling['mean_mm_s']/scaling['std_mm_s']),seqs=[np.arange(n)])

def load_fold(reference,number):
 manifest=json.loads((reference/'split_manifest.json').read_text(encoding='utf-8-sig'))
 if manifest.get('velocity_method')!='LinearFit5':raise ValueError('Expected D1 LinearFit5 reference')
 data=Path(manifest['source']);split=manifest['folds'][number-1];groups=[]
 for label in ['Train','Test']:
  group=[]
  for r in split[label]:
   folder=data/f"{r['size_cm']*10}mm"/r['trial']
   if base.file_hash(folder/'camera.csv')!=r['csv_sha256']:raise ValueError(f'Reference data changed: {folder}')
   g=base.load_trial(folder);g['size_cm']=r['size_cm'];g['use_velocity']=True;group.append(g)
  groups.append(group)
 assert not {(g['size_cm'],g['name']) for g in groups[0]}&{(g['size_cm'],g['name']) for g in groups[1]}
 def source_frames(group):return {(int(f),float(t)) for g in group for f,t in zip(g['frame'],g['unix'])}
 if source_frames(groups[0])&source_frames(groups[1]):raise ValueError('Train/test source frames overlap')
 return manifest,groups[0],groups[1]

@torch.inference_mode()
def evaluate(model,test,mean,std,device,out):
 records=[]
 for g in test:
  p=np.full(len(g['y']),np.nan)
  for ids,x in base.batches(g,'LSTM Full',device):p[ids]=model(x).reshape(-1).cpu().numpy()*std+mean
  ids=g['common'];records.append(dict(size_cm=g['size_cm'],trial=g['name'],model='LSTM Full',**base.scores(g['y'][ids],p[ids])))
  base.write_csv(out/'Test'/f"{g['size_cm']}cm"/f"{g['name']}.csv",[dict(time_s=float(g['t'][i]),actual_mm=float(g['y'][i]),lstm_full_mm=float(p[i]),common_valid=int(i in set(ids))) for i in range(len(p))])
 base.write_csv(out/'test_per_trial_metrics.csv',records)
 base.write_csv(out/'test_mean_sd.csv',[r for r in base.summarize(records) if r['model']=='LSTM Full'])
 return records

@torch.inference_mode()
def static_diagnostic(model,mean,std,scaling,device,out):
 """Independent long constant/noisy input checks; not used to select checkpoints."""
 records=[];traces=[];rng=np.random.default_rng(20261124)
 for size in [2,3,4]:
  for mode in ['constant','distance_noise_0p2mm']:
   seconds=30;times=np.arange(seconds*30)/30;distance=np.full(len(times),300.)
   if mode!='constant':distance+=rng.normal(0,.2,len(times));distance[0]=300.
   velocity=base.estimate_velocity(distance,times,[np.arange(len(times))]);idx=np.arange(4,len(times))
   x=np.column_stack([distance[idx]/distance[0],np.full(len(idx),size),(velocity[idx]-scaling['mean_mm_s'])/scaling['std_mm_s']])
   tensor=torch.as_tensor(x,dtype=torch.float32,device=device).unsqueeze(0);values=model(tensor).reshape(-1).cpu().numpy()*std+mean
   start=float(values[0]);records.append(dict(size_cm=size,input=mode,duration_s=seconds,initial_mm=start,final_mm=float(values[-1]),max_mm=float(values.max()),min_mm=float(values.min()),max_abs_drift_mm=float(np.max(np.abs(values-start))),sd_mm=float(values.std(ddof=1))))
   traces.extend(dict(size_cm=size,input=mode,time_s=float(t),prediction_mm=float(v)) for t,v in zip(times[idx],values))
 base.write_csv(out/'static_stability_summary.csv',records);base.write_csv(out/'static_stability_traces.csv',traces)
 for r in records:print(f"Rest check {r['size_cm']*10} mm / {r['input']}: max drift {r['max_abs_drift_mm']:.2f} mm",flush=True)


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--reference',type=Path,default=REFERENCE)
 parser.add_argument('--fold',type=int,choices=[1,2,3,4,5],default=1)
 parser.add_argument('--epochs',type=int,default=250)
 parser.add_argument('--static-seconds',type=float,nargs='+',default=[0,.5,1,2,3,5])
 parser.add_argument('--rest-only-prob',type=float,default=.2,help='Probability of an additional rest-only training update per trial')
 parser.add_argument('--rest-only-seconds',type=float,nargs='+',default=[1,3,5,10])
 parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
 parser.add_argument('--seed',type=int,default=20261029)
 parser.add_argument('--check-data',action='store_true')
 parser.add_argument('--smoke-test',action='store_true',help='One temporary training update; no trained result is saved')
 args=parser.parse_args()
 if args.epochs<1 or not 0<=args.rest_only_prob<=1 or any(not np.isfinite(v) or v<0 for v in args.static_seconds) or any(not np.isfinite(v) or v<=0 for v in args.rest_only_seconds):parser.error('Invalid epochs, probability or durations')
 if args.device=='auto':args.device='cuda' if torch.cuda.is_available() else 'cpu'
 if args.device=='cuda' and not torch.cuda.is_available():parser.error('CUDA is unavailable')
 torch.set_num_threads(min(4,torch.get_num_threads()));torch.manual_seed(args.seed);np.random.seed(args.seed)
 manifest,train,test=load_fold(args.reference,args.fold);folder=args.reference/f'fold_{args.fold:02d}'
 spec=json.loads((folder/'function.json').read_text());scaling=spec['velocity_scaling'];old=torch.load(folder/'lstm_full.pth',map_location='cpu',weights_only=False)
 if old['velocity_scaling']!=scaling:raise ValueError('Velocity scaler mismatch')
 mean,std=float(old['mean_mm']),float(old['std_mm'])
 for g in train+test:g['velocity_scaled']=(g['velocity_mm_s']-scaling['mean_mm_s'])/scaling['std_mm_s']
 print(f'D1 fold {args.fold}: {len(train)} train / {len(test)} untouched test trials',flush=True)
 print('LinearFit5; retrain LSTM from scratch; preserve Function/MLP and their scalers.',flush=True)
 for size in [2,3,4]:
  g=next(g for g in train if g['size_cm']==size and g['seqs'][0][0]==4)
  for seconds in args.static_seconds:
   padded,n=prepend_rest(g,seconds,scaling)
   if n:
    np.testing.assert_array_equal(padded['y'][n:],g['y']);np.testing.assert_allclose(padded['d'][n:],g['d']);np.testing.assert_allclose(padded['velocity_mm_s'][4:n],0,atol=1e-8)
   for ids,x in base.batches(padded,'LSTM Full','cpu'):assert torch.isfinite(x).all()
  static=rest_only(g,10,scaling);assert np.all(static['d']==1) and np.all(static['velocity_mm_s']==0)
  np.testing.assert_allclose(static['velocity_scaled'],-scaling['mean_mm_s']/scaling['std_mm_s'])
 if args.check_data:
  print('PASS: fold/hash integrity, disjoint original trials, causal inputs, unchanged original targets/L0, finite rest prefixes and correctly scaled zero velocity. No training started.',flush=True);return
 model=base.JointLSTM(True).to(args.device);optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
 def update(g):
  optimizer.zero_grad(set_to_none=True);errors=[]
  for ids,x in base.batches(g,'LSTM Full',args.device):
   y=torch.as_tensor((g['y'][ids]-mean)/std,dtype=torch.float32,device=args.device);errors.append((model(x).reshape(-1)-y).abs())
  loss=torch.cat(errors).mean();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();return float(loss.detach())*std
 if args.smoke_test:
  model.train();g,_=prepend_rest(train[0],1,scaling);loss=update(g);loss2=update(rest_only(train[0],3,scaling));assert np.isfinite(loss+loss2);print('PASS: one padded-motion and one rest-only training update, no checkpoint saved.',flush=True);return
 out=Path(manifest['source'])/'Result'/f'LSTM_LinearFit5_Static_D1_fold{args.fold:02d}_{datetime.now():%Y%m%d_%H%M%S}'
 out.mkdir(parents=True,exist_ok=False)
 for file in ['function.json','mlp5.pth','velocity_scaling.json']:
  if (folder/file).exists():shutil.copy2(folder/file,out/file)
 config=dict(reference=str(args.reference),reference_fold=args.fold,source=manifest['source'],epochs=args.epochs,seed=args.seed,velocity_method='LinearFit5',static_prefix_seconds=args.static_seconds,static_prefix_resampled_each_epoch=True,rest_only_probability=args.rest_only_prob,rest_only_seconds=args.rest_only_seconds,rest_target='median first five measured aperture frames',rest_distance='original initial distance; original L0 preserved',rest_velocity='physical zero, transformed by unchanged reference scaler',function_and_mlp='copied unchanged; MLP live single-pass is independent of LSTM training',evaluation='single fixed held-out fold, NOT a five-fold aggregate',initialization='LSTM from scratch',Train=manifest['folds'][args.fold-1]['Train'],Test=manifest['folds'][args.fold-1]['Test'])
 base.write_json(out/'training_config.json',config);print('Output:',out,flush=True)
 history=[];augmentation_log=[];started=time.perf_counter()
 for epoch in range(1,args.epochs+1):
  model.train();rng=random.Random(args.seed+epoch);order=list(train);rng.shuffle(order);motion_loss=[];static_loss=[];counts=0
  for g in order:
   seconds=rng.choice(args.static_seconds);aug,n=prepend_rest(g,seconds,scaling);motion_loss.append(update(aug));counts+=int(n>0)
   augmentation_log.append(dict(epoch=epoch,trial=g['name'],size_cm=g['size_cm'],requested_seconds=seconds,added_frames=n))
   if rng.random()<args.rest_only_prob:static_loss.append(update(rest_only(g,rng.choice(args.rest_only_seconds),scaling)))
  history.append(dict(epoch=epoch,motion_train_mae_mm=float(np.mean(motion_loss)),rest_only_train_mae_mm=float(np.mean(static_loss)) if static_loss else None,padded_trials=counts,rest_only_sequences=len(static_loss),elapsed_minutes=(time.perf_counter()-started)/60))
  if epoch==1 or epoch%10==0 or epoch==args.epochs:
   h=history[-1];print(f"Epoch {epoch}/{args.epochs} | motion {h['motion_train_mae_mm']:.3f} mm | rest {h['rest_only_train_mae_mm']:.3f} mm | elapsed {h['elapsed_minutes']:.1f} min",flush=True);base.write_csv(out/'training_history.csv',history)
  if epoch%25==0 or epoch==args.epochs:
   torch.save(dict(state_dict=model.state_dict(),mean_mm=mean,std_mm=std,epochs=epoch,velocity_method='LinearFit5',velocity_scaling=scaling,static_training=config),out/'lstm_full.pth')
 base.write_csv(out/'static_augmentation_log.csv',augmentation_log);model.eval()
 evaluate(model,test,mean,std,args.device,out);static_diagnostic(model,mean,std,scaling,args.device,out)
 base.write_json(out/'completion.json',dict(complete=True,epochs=args.epochs,test_trials=len(test),static_tests='30 s constant and small distance-noise input, three sizes',note='Inspect static_stability_summary.csv before deployment. No output clamping/gating or hidden state resets were used to hide drift.'))
 print('Saved complete bundle:',out,flush=True)
if __name__=='__main__':main()
