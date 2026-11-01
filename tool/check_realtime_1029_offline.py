import sys,csv,json,time
from pathlib import Path
sys.path.insert(0,str(Path('New/Program').resolve()))
import realtime_1029_d1_l0_function_mlp_lstm as m
import numpy as np
m.torch.set_num_threads(1)
results=[]
for obj in ['20mm_red','30mm_blue','40mm_black']:
 base=m.ROOT/'Data/1029_D1'/obj;run=sorted(p for p in (base/'Result').glob('l0_normalized_function_mlp5_lstmfull_400ep_*') if (p/'models/lstm_full.pth').exists())[-1]
 e=m.ComparisonEngine(run/'models');m.check_models(e);e.reset()
 p=sorted((base/'Test').glob('grasp_*/camera.csv'))[0]
 with p.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 expected_file=run/'Test/predictions'/f'{p.parent.name}.csv'
 with expected_file.open(encoding='utf-8-sig',newline='') as f:expected={int(r['source_frame']):r for r in csv.DictReader(f)}
 compared=0;diff=0;export=[]
 for i,r in enumerate(rows):
  valid=r['valid']=='1' and r['index_mcp_distance_valid']=='1'
  prediction,common,gap=e.step(float(r['index_mcp_object_distance_mm']) if valid else None,float(r['t_camera']),int(r['frame']))
  if common:
   ex=expected[int(r['frame'])]
   for key,savedkey in zip(m.KEYS,['function_mm','mlp_5f_mm','lstm_full_mm']):diff=max(diff,abs(prediction[key]-float(ex[savedkey])))
   compared+=1
  export.append(dict(t_camera=float(r['t_camera']),source_frame=int(r['frame']),valid=valid,finger_width_mm=float(r['finger_width_mm']),common_valid=common,reset_on_gap=gap,inference_ms=sum(v for v in e.last_timing.values() if np.isfinite(v)),detection_ms=0.,**prediction))
 assert compared>0 and diff<.02,(obj,diff)
 session=m.ROOT/'New/Result/RealTime_1029_D1'/('OFFLINE_EXPORT_CHECK_'+obj+'_'+str(time.time_ns()));session.mkdir(parents=True)
 image=np.zeros((380,540,3),np.uint8)
 trial=m.LiveTrial(session,image.shape,30,dict(synthetic_black_video=True,source=str(p),note='Offline export check; no camera'))
 for row in export:trial.append(row,image,image)
 trial.close();chart=m.chart(trial.rows);m.cv2.imwrite(str(session/'chart_check.png'),chart)
 for name in ['raw_video.mp4','annotated_video.mp4']:
  cap=m.cv2.VideoCapture(str(trial.path/name));count=int(cap.get(m.cv2.CAP_PROP_FRAME_COUNT));cap.release();assert count==len(rows),(name,count,len(rows))
 results.append(dict(object=obj,frames=len(rows),common_frames=compared,max_difference_mm=diff,output=str(session)))
print(json.dumps(results,indent=2))
