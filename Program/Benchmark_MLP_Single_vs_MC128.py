"""Benchmark deterministic single-pass MLP versus MC Dropout x128 on saved real inputs."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
import torch
from Train_D1_Joint_Three_Models_LinearFit5_CV5 import JointMLP,load_trial,batches
ROOT=Path(__file__).resolve().parents[1]
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--models',type=Path,default=ROOT/'Data/1101_D_PreAction15/Result/Joint_Three_Models_LinearFit5_CurrentV_CV5_20261123_153838/fold_01')
 parser.add_argument('--repeats',type=int,default=2000)
 args=parser.parse_args()
 if args.repeats<100:parser.error('Use at least 100 repetitions')
 torch.set_num_threads(1);torch.manual_seed(20261029)
 ck=torch.load(args.models/'mlp5.pth',map_location='cpu',weights_only=False)
 model=JointMLP(True,ck.get('mlp_hidden_width',32));model.load_state_dict(ck['state_dict']);model.eval()
 source=next((args.models/'Test/2cm').glob('*.csv'))
 with source.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
 valid=[r for r in rows if r['common_valid']=='1'];row=valid[len(valid)//2]
 # Use the input time history from the actual training dataset and its fold-specific scaler.
 dataset=ROOT/'Data/1101_D_PreAction15'
 if '1115_Song' in str(args.models):dataset=ROOT/'Data/1115_Song'
 folder=dataset/('20' if dataset.name=='1115_Song' else '20mm')/source.stem
 trial=load_trial(folder);trial['size_cm']=2;trial['use_velocity']=True
 scale=ck['velocity_scaling'];trial['velocity_scaled']=(trial['velocity_mm_s']-scale['mean_mm_s'])/scale['std_mm_s']
 ids,features=next(batches(trial,'MLP 5f','cpu'));inputs=features.numpy()
 timings={k:[] for k in ['Single pass (Dropout off)','MC Dropout x128 (median)']}
 predictions={k:[] for k in timings}
 with torch.inference_mode():
  for i in range(args.repeats+200):
   values=inputs[i%len(inputs)]
   modes=list(timings)
   if i%2:modes.reverse()
   for mode in modes:
    mc=mode.startswith('MC');model.net[2].train(mc)
    began=time.perf_counter_ns()
    x=torch.tensor(values,dtype=torch.float32)
    if mc:result=torch.quantile(model(x.expand(128,-1)),.5).item()
    else:result=model(x.unsqueeze(0)).item()
    value=result*ck['std_mm']+ck['mean_mm']
    elapsed=(time.perf_counter_ns()-began)/1e6
    if i>=200:timings[mode].append(elapsed)
 summary=[]
 for mode,values in timings.items():
  a=np.array(values);r=dict(mode=mode,n=len(a),mean_ms=float(a.mean()),sd_ms=float(a.std(ddof=1)),median_ms=float(np.median(a)),p95_ms=float(np.percentile(a,95)))
  summary.append(r);print(r,flush=True)
 out=ROOT/'Report/MLP_Inference_Benchmark';out.mkdir(exist_ok=True)
 with (out/'timing_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 (out/'benchmark_metadata.json').write_text(json.dumps(dict(checkpoint=str(args.models/'mlp5.pth'),real_input_source=str(source),threads=1,device='cpu',warmup_per_mode=200,timing_scope='feature tensor construction, forward pass, reduction, output unscaling; excludes camera/detection; dropout mode switch excluded',torch_version=torch.__version__,note='Other training jobs may affect timings; rerun when idle. Training weights unchanged; not a no-dropout-trained model.'),indent=2),encoding='utf-8')
 print('Saved',out)
if __name__=='__main__':main()
