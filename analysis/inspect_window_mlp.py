from pathlib import Path
import sys,json,time,platform
import numpy as np
import pandas as pd
import torch
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
import train_window_mlp as m
result=root/'Data/Result/singleSize_20261013_173723_228733'
split=pd.read_csv(result/'data_split.csv')
trials=m.load_trials(m.DATA_ROOT)
report=dict(torch_version=torch.__version__,python_version=platform.python_version(),cuda_available=torch.cuda.is_available(),
            torch_threads=torch.get_num_threads(),models=[])
if torch.cuda.is_available():report['gpu']=torch.cuda.get_device_name(0)
for name,cols in m.MODEL_SPECS.items():
    path=result/'models'/f'{name}.pth'
    cp=torch.load(path,map_location='cpu',weights_only=False)
    model=m.WindowMLP(cp['window_size'],cp['feature_count'],cp['hidden_size']);model.load_state_dict(cp['state_dict']);model.eval()
    params=sum(p.numel() for p in model.parameters())
    macs=sum(layer.in_features*layer.out_features for layer in model.net if isinstance(layer,torch.nn.Linear))
    x=torch.ones(1,cp['window_size'],cp['feature_count'],dtype=torch.float32)
    mean=cp['x_mean'];scale=cp['x_scale']
    def run():return model((x-mean)/scale)*cp['y_scale']+cp['y_mean']
    timings={}
    for threads in sorted(set([1,report['torch_threads']])):
        torch.set_num_threads(threads)
        with torch.inference_mode():
            for _ in range(100):run()
            times=[]
            for _ in range(2000):
                t=time.perf_counter_ns();run();times.append((time.perf_counter_ns()-t)/1e6)
        timings[str(threads)]=dict(median_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),mean_ms=float(np.mean(times)))
    torch.set_num_threads(report['torch_threads'])
    train_names=split.loc[split.split=='train','trial'].tolist()
    wx,wy=m.make_windows(trials,train_names,cols,cp['window_size'])
    info=dict(name=name,shape=[1,cp['window_size'],cp['feature_count']],parameters=params,
              weight_bytes=params*4,checkpoint_bytes=path.stat().st_size,macs=macs,
              train_windows=len(wx),checkpoint_train_windows=cp['train_windows'],train_trials=len(cp['train_trials']),
              epochs=cp['epochs'],cpu_inference=timings,layers=str(model.net))
    report['models'].append(info)
report['data']=dict(trials=len(trials),rows_min=min(len(d) for d in trials.values()),rows_max=max(len(d) for d in trials.values()),
                    test_trials=int((split.split=='test').sum()))
(result/'model_information.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
