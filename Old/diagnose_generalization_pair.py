from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import train_window_mlp as m
base=Path('Data/Exprement Data/D11')
result=base/'Result/generalization_20261014_152555_886210'
out=result/'pair_diagnosis'
out.mkdir(exist_ok=True)
names=['TRIAL_Size_4.5cm_20261013_110339','TRIAL_Size_4.5cm_20261013_110412']
data=[]
for name in names:
    d=pd.read_csv(base/'4.5'/name/'grasp_data_smoothed.csv')
    d['acceleration_magnitude']=np.linalg.norm(d[['ax','ay','az']],axis=1)
    data.append(d)
models={}
for name,cols in m.MODEL_SPECS.items():
    cp=torch.load(result/'models'/f'{name}.pth',map_location='cpu',weights_only=False)
    net=m.WindowMLP(cp['window_size'],cp['feature_count'],cp['hidden_size']);net.load_state_dict(cp['state_dict']);net.eval()
    models[name]=(cp,net)
def predict(d,name):
    cp,net=models[name];x,y,f=m.make_trial_windows(d,cp['feature_columns'],cp['window_size'])
    with torch.inference_mode():
        p=net(torch.from_numpy((x-cp['x_mean'].numpy())/cp['x_scale'].numpy())).numpy()*cp['y_scale']+cp['y_mean']
    return y,p,f
def progress(d):return 100*(d.frame.to_numpy()-d.frame.iloc[0])/(d.frame.iloc[-1]-d.frame.iloc[0])
grid=np.linspace(0,100,201)
fig,axs=plt.subplots(3,1,figsize=(12,10),sharex=True)
for d,name,color in zip(data,names,['tab:red','tab:blue']):
    for ax,col in zip(axs,['finger_width','object_area','acceleration_magnitude']):ax.plot(progress(d),d[col],label=name[-6:],color=color)
for ax,label in zip(axs,['FW (px)','OA (px2)','Acceleration (m/s2)']):ax.set_ylabel(label);ax.legend();ax.grid(alpha=.2)
axs[-1].set_xlabel('Trial progress (%)');fig.tight_layout();fig.savefig(out/'input_pair.png',dpi=150);plt.close(fig)
rows=[]
for i,d in enumerate(data):
    print(names[i], 'rows',len(d),'seconds',d.t_camera.iloc[-1]-d.t_camera.iloc[0] if 't_camera' in d else None)
    for col in ['finger_width','object_area','acceleration_magnitude']:
        print(col,'start',d[col].iloc[:5].mean(),'end',d[col].iloc[-5:].mean(),'peak',d[col].max(),'at',progress(d)[d[col].argmax()])
    for model in models:
        y,p,f=predict(d,model);x=100*(f-d.frame.iloc[0])/(d.frame.iloc[-1]-d.frame.iloc[0]);print('BASE',model,m.regression_metrics(y,p))
        for lo,hi in [(0,40),(40,60),(60,80),(80,101)]:
            mask=(x>=lo)&(x<hi);rows.append(dict(trial=names[i],model=model,stage=f'{lo}-{min(hi,100)}',rmse=np.sqrt(np.mean((p[mask]-y[mask])**2)),bias=np.mean(p[mask]-y[mask])))
    other=data[1-i]
    for col in ['object_area','acceleration_magnitude']:
        alternative=d.copy();alternative[col]=np.interp(progress(d),progress(other),other[col])
        y,p,f=predict(alternative,'window_area_acceleration');print('SWAP',names[i][-6:],col,'RMSE',m.regression_metrics(y,p)[0])
    for col in ['object_area','acceleration_magnitude']:
        scores=[]
        for shift in range(-8,9):
            alternative=d.copy();alternative[col]=np.interp(np.arange(len(d))+shift,np.arange(len(d)),d[col])
            y,p,f=predict(alternative,'window_area_acceleration');scores.append((m.regression_metrics(y,p)[0],shift))
        print('SHIFT BEST',names[i][-6:],col,min(scores))
pd.DataFrame(rows).to_csv(out/'stage_errors.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
