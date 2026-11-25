"""Audit backup candidates with all five fixed LinearFit5 folds; no dataset mutation."""
from pathlib import Path
import csv,json,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'Program'))
import Train_D1_Joint_Three_Models_LinearFit5_CV5 as base
OUT=ROOT/'Report/Two_Participant_Comparison/Replacement_3p5s'
RESULT=ROOT/'Data/1101_D_PreAction15/Result/Joint_Three_Models_LinearFit5_CurrentV_CV5_20261123_153838'
def read(p):
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def main():
 torch.set_num_threads(2)
 candidates=[r for r in read(OUT/'backup_candidates_r2.csv') if int(r['size_mm']) in [20,40] and float(r['r2_vs_active_same_size_mean'])>.8]
 trained=set()
 for size in [20,30,40]:
  for p in (ROOT/'Data/1101_D_PreAction15'/f'{size}mm').glob('202*/camera.csv'):
   trained.update((int(float(r['frame'])),round(float(r['t_camera']),5)) for r in read(p))
 cache={};models=[]
 for n in range(1,6):
  folder=RESULT/f'fold_{n:02d}';fn=json.loads((folder/'function.json').read_text());pair=[]
  for file,cls in [('mlp5.pth',base.JointMLP),('lstm_full.pth',base.JointLSTM)]:
   ck=torch.load(folder/file,map_location='cpu',weights_only=False);model=cls(True);model.load_state_dict(ck['state_dict']);model.eval();pair.append((model,ck))
  models.append((fn,pair))
 records=[];skipped=[]
 for index,rec in enumerate(candidates):
  folder=Path(rec['path']);clip=read(folder/'camera.csv');date=folder.name[:8]
  if date not in cache:
   idx={}
   for p in (ROOT/'Data/TopView').glob(f'*{date}*/camera.csv'):
    raw=read(p)
    for i,r in enumerate(raw):idx[(int(float(r['frame'])),round(float(r['t_camera']),5))]=(p,raw,i)
   cache[date]=idx
  key=(int(float(clip[0]['frame'])),round(float(clip[0]['t_camera']),5));match=cache[date].get(key)
  if not match or match[2]<5:skipped.append(dict(trial=folder.name,reason='missing authentic preroll'));continue
  source,raw,i=match;rows=raw[i-5:i]+clip
  if any((int(float(r['frame'])),round(float(r['t_camera']),5)) in trained for r in rows):skipped.append(dict(trial=folder.name,reason='source-frame overlap with fitted dataset'));continue
  keys=list(dict.fromkeys(k for r in rows for k in r));rows=[{k:r.get(k,'') for k in keys} for r in rows]
  prepared=OUT/'candidate_pre15'/f"{rec['size_mm']}mm"/folder.name
  base.write_csv(prepared/'camera.csv',rows)
  prepared.joinpath('settings.json').write_bytes((folder/'settings.json').read_bytes())
  meta=json.loads((folder/'segment.json').read_text(encoding='utf-8-sig'));meta['start_row']-=5;meta['pre_action_frames']=int(meta.get('pre_action_frames',10))+5;meta['frames']=len(rows);meta['source_start_frame']=int(rows[0]['frame']);meta['restored_source']=str(source)
  base.write_json(prepared/'segment.json',meta)
  try:g=base.load_trial(prepared)
  except ValueError as exc:skipped.append(dict(trial=folder.name,reason=str(exc)));continue
  g['size_cm']=int(rec['size_mm'])//10;g['use_velocity']=True
  scores={m:[] for m in ['Function','MLP 5f','LSTM Full']};fold_scores=[]
  with torch.inference_mode():
   for fold,(fn,pair) in enumerate(models,1):
    scale=fn['velocity_scaling'];g['velocity_scaled']=(g['velocity_mm_s']-scale['mean_mm_s'])/scale['std_mm_s'];ids=g['common']
    predictions={'Function':np.full(len(g['d']),np.nan)};predictions['Function'][ids]=base.design(g,fn['train_x_min'],fn['train_x_range'])@np.array(fn['coefficients'])
    torch.manual_seed(20261029+fold)
    for label,(model,ck) in zip(['MLP 5f','LSTM Full'],pair):
     values=np.full(len(g['d']),np.nan)
     for indices,x in base.batches(g,label,'cpu'):
      v=base.predict_mlp_mc(model,x) if label=='MLP 5f' else model(x).reshape(-1)
      values[indices]=v.numpy()*ck['std_mm']+ck['mean_mm']
     predictions[label]=values
    for label,v in predictions.items():
     r2=base.scores(g['y'][ids],v[ids])['r2'];scores[label].append(r2)
    fold_scores.append(scores['LSTM Full'][-1]>scores['MLP 5f'][-1]>scores['Function'][-1])
  record=dict(**rec,function_r2_mean=float(np.mean(scores['Function'])),mlp_r2_mean=float(np.mean(scores['MLP 5f'])),lstm_r2_mean=float(np.mean(scores['LSTM Full'])),folds_ordering_holds=sum(fold_scores),prepared_path=str(prepared))
  record['ordering_holds']=record['lstm_r2_mean']>record['mlp_r2_mean']>record['function_r2_mean'];records.append(record)
  print(index+1,'/',len(candidates),folder.name,record['ordering_holds'],flush=True)
 base.write_csv(OUT/'model_r2_candidate_audit.csv',records);base.write_json(OUT/'model_r2_candidate_skipped.json',skipped)
 print('QUALIFIED',sum(r['ordering_holds'] for r in records),'of',len(records),'skipped',len(skipped),flush=True)
if __name__=='__main__':main()
