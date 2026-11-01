"""Benchmark saved models with causal, single-frame offline replay; no training/camera.
Run with the same Python environment used for deployment.
"""
from __future__ import annotations
import argparse,csv,json,time,hashlib,platform
from pathlib import Path
from collections import deque
import numpy as np
import torch
from scipy.interpolate import BSpline
from infer_1029_d1_slow_grasp import MLP,FullLSTM
BASE=Path(__file__).resolve().parents[2]

def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(values):
 a=np.asarray(values);return dict(n=len(a),mean_ms=float(a.mean()),median_ms=float(np.median(a)),p95_ms=float(np.percentile(a,95)),p99_ms=float(np.percentile(a,99)),max_ms=float(a.max()),equivalent_fps=1000/float(a.mean()),over_30fps_budget_pct=float(100*np.mean(a>1000/30)))
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');ap.add_argument('--threads',type=int,default=1);ap.add_argument('--repeats',type=int,default=2);ap.add_argument('--warmup',type=int,default=50);ap.add_argument('--objects',nargs='+',default=['20mm_red','30mm_blue','40mm_black']);args=ap.parse_args()
 if args.threads<1 or args.repeats<1 or args.warmup<1:ap.error('threads/repeats/warmup must be positive')
 if args.device=='cuda' and not torch.cuda.is_available():ap.error('CUDA unavailable')
 torch.set_num_threads(args.threads);torch.set_num_interop_threads(1);device=torch.device(args.device)
 out=BASE/'Data/1029_D1/realtime_benchmark'/time.strftime('%Y%m%d_%H%M%S');out.mkdir(parents=True,exist_ok=False)
 def sync():
  if device.type=='cuda':torch.cuda.synchronize(device)
 all_rows=[];summary=[];verification=[];hashes={};loads=[]
 with torch.inference_mode():
  for obj in args.objects:
   data=BASE/'Data/1029_D1'/obj
   runs=sorted(p for p in (data/'Result').glob('l0_normalized_function_mlp5_lstmfull_400ep_*') if (p/'models/lstm_full.pth').is_file());assert runs,f'No trained run: {obj}'
   run=runs[-1];files=[run/'models'/n for n in ['function.json','mlp_5f.pth','lstm_full.pth']]
   for p in files:hashes[str(p)]=sha(p)
   start=time.perf_counter();f=json.loads(files[0].read_text(encoding='utf-8-sig'));mc=torch.load(files[1],map_location='cpu',weights_only=True);lc=torch.load(files[2],map_location='cpu',weights_only=True)
   for ck in [f,mc,lc]:
    assert ck.get('input_column')=='index_mcp_object_distance_mm' and ck.get('target_column')=='finger_width_mm'
    assert not ck.get('simulated_untrained') and 'first recorded frame' in ck.get('physical_normalization','')
   assert mc['epochs']==lc['epochs']==400
   scale=mc['numeric_scaler'];assert f['numeric_scaler']==lc['numeric_scaler']==scale
   spec=f['definition'];spline=BSpline(spec['knots'],spec['coefficients'],spec['degree'],extrapolate=True)
   mlp=MLP(mc.get('hidden_size',32)).to(device);mlp.load_state_dict(mc['state_dict']);mlp.eval()
   lstm=FullLSTM(lc.get('hidden_size',32)).to(device);lstm.load_state_dict(lc['state_dict']);lstm.eval();sync()
   loads.append(dict(object=obj,run=str(run),model_load_s=time.perf_counter()-start))
   def tensor(x):return torch.tensor(x,dtype=torch.float32,device=device)
   # Warm up each model independently; reset state before each real trial.
   for _ in range(args.warmup):
    spline(.5);mlp(tensor([[.5]*5])).cpu().item();estimate,_=lstm(tensor([[[.5]]]),None);estimate.cpu().item()
   sync();object_rows=[];maxgap=float(mc.get('max_frame_gap_s',.1));trial_count=0
   for p in sorted((data/'Test').glob('grasp_*/camera.csv')):
    hashes[str(p)]=sha(p);settings=p.parent/'settings.json';hashes[str(settings)]=sha(settings)
    calibration=json.loads(settings.read_text(encoding='utf-8-sig')).get('pixel_calibration',{}).get('mm_per_px');assert calibration is not None and float(calibration)>0
    rows=read(p)
    def num(r,k):
     try:return float(r[k])
     except (ValueError,TypeError,KeyError):return float('nan')
    d=np.array([num(r,'index_mcp_object_distance_mm') for r in rows]);t=np.array([num(r,'t_camera') for r in rows]);frames=np.array([num(r,'frame') for r in rows]);valid=np.array([r.get('valid')=='1' and r.get('index_mcp_distance_valid')=='1' for r in rows])&np.isfinite(d)&np.isfinite(t)&np.isfinite(frames)
    assert valid[0] and d[0]>0;dt=np.diff(t);budget_ms=float(np.median(dt[(dt>0)&(dt<=maxgap)])*1000);trial_count+=1
    # One correctness check per object: batch full history versus carried single-step state.
    if trial_count==1:
     stop=1
     while stop<len(d) and valid[stop] and frames[stop]-frames[stop-1]==1 and 0<t[stop]-t[stop-1]<=maxgap:stop+=1
     xs=(d[:stop]/d[0]-scale['x_mean'])/scale['x_std'];batch,_=lstm(tensor(xs).reshape(1,-1,1));state=None;online=[]
     for x in xs:
      y,state=lstm(tensor([[[x]]]),state);online.append(y.cpu().item())
     diff=float(np.max(np.abs(batch.cpu().numpy().ravel()-online))*scale['y_std']);assert diff<.01,(obj,diff)
     verification.append(dict(object=obj,frames=stop,max_incremental_vs_full_sequence_difference_mm=diff))
    for repeat in range(args.repeats):
     for label in ['Function','MLP 5f','LSTM Full']:
      state=None;window=deque(maxlen=5);count=0;previous=None
      for i in range(len(rows)):
       if not valid[i]:state=None;window.clear();count=0;previous=None;continue
       if previous is not None and (frames[i]-frames[previous]!=1 or not 0<t[i]-t[previous]<=maxgap):state=None;window.clear();count=0
       # Timer includes normalization, rolling window/state, tensor creation/transfer,
       # forward pass, output transfer and conversion back to millimetres.
       sync();started=time.perf_counter_ns();x=(float(d[i])/float(d[0])-scale['x_mean'])/scale['x_std'];window.append(x);count+=1
       if label=='Function':value=float(spline(x))
       elif label=='MLP 5f':
        if count<5:previous=i;continue
        value=mlp(tensor([list(window)])).cpu().item()
       else:
        y,state=lstm(tensor([[[x]]]),state);value=y.cpu().item()
       value=value*scale['y_std']+scale['y_mean'];sync();elapsed=(time.perf_counter_ns()-started)/1e6
       assert np.isfinite(value)
       object_rows.append(dict(object=obj,trial=p.parent.name,repeat=repeat,model=label,source_frame=int(frames[i]),common_valid=int(count>=5),latency_ms=elapsed,recorded_frame_budget_ms=budget_ms,prediction_mm=value));previous=i
   for label in ['Function','MLP 5f','LSTM Full']:
    samples=[r for r in object_rows if r['model']==label and r['common_valid']];s=dict(object=obj,model=label,test_trials=trial_count,**stats([r['latency_ms'] for r in samples]));s['over_recorded_budget_pct']=100*float(np.mean([r['latency_ms']>r['recorded_frame_budget_ms'] for r in samples]));s['median_recorded_frame_budget_ms']=float(np.median([r['recorded_frame_budget_ms'] for r in samples]));summary.append(s)
   all_rows+=object_rows;print(json.dumps([r for r in summary if r['object']==obj]),flush=True)
 assert all(sha(Path(p))==h for p,h in hashes.items()),'Source/model changed'
 write(out/'per_frame_latency.csv',all_rows);write(out/'summary.csv',summary)
 report=dict(scope='Offline model inference pipeline only. Excludes camera capture, hand/object detection, drawing, robot control, disk IO and playback sleeping. Models timed separately, not concurrently. CUDA timings synchronized including transfers. Equivalent FPS is 1000/mean latency, not measured camera FPS.',fair_comparison='Summary uses frames with >=5 contiguous valid inputs; LSTM still consumes initial frames. Raw timings also include LSTM start frames. Trial/gap resets; no future input; normalization uses first recorded distance.',environment=dict(python=platform.python_version(),torch=torch.__version__,processor=platform.processor(),device=str(device),cuda_name=torch.cuda.get_device_name(device) if device.type=='cuda' else None,threads=args.threads,repeats=args.repeats,warmup=args.warmup),models=loads,correctness=verification,summary=summary,source_hashes=hashes,source_files_unchanged=True)
 (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print('Saved:',out,flush=True)
if __name__=='__main__':main()
