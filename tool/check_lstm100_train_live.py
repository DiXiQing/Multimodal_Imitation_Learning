import sys
from pathlib import Path
sys.path.insert(0,str(Path('New/Program').resolve()))
import train_1031_d_zscore_lstm100 as train
import realtime_1031_d_zscore_lstm100 as live
import numpy as np,torch
folder=Path('Data/1031_D/30mm/Result/CHECK_ONLY_ZSCORE_UNTRAINED_20261031_194305_559365/models')
torch.set_num_threads(1)
e=live.ComparisonEngine(folder,allow_simulated=True)
m=train.Window100LSTM();m.load_state_dict(e.lstm.state_dict());m.eval()
d=np.linspace(360,35,155);x=(d-e.scale['x_mean'])/e.scale['x_std']
with torch.inference_mode():pred,_=m(torch.tensor(x,dtype=torch.float32).reshape(1,-1,1))
expected=pred.numpy()[0]*e.scale['y_std']+e.scale['y_mean'];got=[]
for i,value in enumerate(d):got.append(e.step(value,100+i/30,i)[0][live.KEYS[2]])
error=float(np.max(abs(expected-got)));assert error<.001,error
try:live.ComparisonEngine(folder)
except ValueError:print('PASS: live mode rejects untrained mock checkpoints')
else:raise AssertionError('Mock checkpoints accepted for camera mode')
print('PASS: packed training windows vs real-time replay, 155 frames, max difference mm:',error)
