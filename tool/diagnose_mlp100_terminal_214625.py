import sys,csv,json,numpy as np,torch
from pathlib import Path
sys.path.insert(0,str(Path('New/Program').resolve()))
import realtime_1031_d_l0_mlp100_lstm5 as live
import train_1031_d_l0_mlp100_lstm5 as train
torch.set_num_threads(1)
session=Path('New/Result/RealTime_1031_L0_MLP100_LSTM5/three_model_live_20261031_214625_682811');source=json.loads((session/'model_sources.json').read_text());e=live.ComparisonEngine(Path(source['models']));sc=e.scale
for ratio in [1.,.16,.17]:
 with torch.inference_mode():y=e.mlp(torch.full((1,100),ratio)).item()*sc['y_std']+sc['y_mean']
 print('CONSTANT 100 distances',ratio,'predicted aperture',y)
windows=[];apertures=[]
for folder in sorted(Path('Data/1031_D/30mm/Train').glob('grasp_*')):
 g=train.load_trial(folder);windows.extend(g['windows']);apertures.extend(g['y'][g['common']])
w=np.array(windows);near=w.mean(axis=1)<.25;flat=np.ptp(w,axis=1)<.03
print('TRAIN windows',len(w),'near-flat windows',int((near&flat).sum()),'near windows',int(near.sum()),'near span min',float(np.ptp(w,axis=1)[near].min()) if near.any() else None)
for f in sorted(session.glob('trial_*/predictions.csv')):
 rows=list(csv.DictReader(f.open(encoding='utf-8-sig')));e.reset();diff=[];starts=[]
 for i,r in enumerate(rows):
  pred,c,_=e.step(float(r['index_mcp_object_distance_mm']) if r['valid']=='True' else None,float(r['t_camera']),int(r['source_frame']))
  saved=float(r['predicted_mlp_mm'])
  if np.isfinite(saved):diff.append(abs(saved-pred[live.KEYS[1]]))
 print('REPLAY',f.parent.name,'max difference',max(diff))
 if f==sorted(session.glob('trial_*/predictions.csv'))[0]:
  for i in range(0,len(rows),30):
   r=rows[i];print('TIMELINE',i,*(round(float(r[k]),2) for k in ['elapsed_s','distance_over_initial','finger_width_mm','predicted_mlp_mm']))
