"""Green-marker live aperture estimation. N start, S stop, click target, Q quit.
Uses current + four past frames (LinearFit5); no hand-landmark measurements.
"""
import argparse,csv,json,time
from collections import deque
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import torch
from scipy.interpolate import BSpline
from Train_D1_Joint_Three_Models_LinearFit5_CV5 import JointMLP,JointLSTM
ROOT=Path(__file__).resolve().parents[1]
BUNDLES={
 'song':ROOT/'Data/1115_Song/Result/Joint_Three_Models_LinearFit5_CurrentV_CV5_20261123_153753/fold_01',
 'd1':ROOT/'Data/1101_D_PreAction15/Result/Joint_Three_Models_LinearFit5_CurrentV_CV5_20261123_153838/fold_01'}
KEYS=['function_mm','mlp_mm','lstm_mm']
COLORS=[(190,125,45),(40,155,245),(75,170,60)]
WINDOW='Green marker | click cube center | N start S stop Q quit'

def put(im,text,xy,color=(235,235,235),scale=.5):
 cv2.putText(im,text,xy,cv2.FONT_HERSHEY_SIMPLEX,scale,(15,15,15),3,cv2.LINE_AA)
 cv2.putText(im,text,xy,cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)

class Engine:
 def __init__(self,folder,size,samples):
  self.folder=folder;self.size=size;self.samples=samples
  self.spec=json.loads((folder/'function.json').read_text(encoding='utf-8-sig'))
  if self.spec.get('velocity_method')!='LinearFit5':raise ValueError('This script requires a LinearFit5 model bundle')
  self.mlp=JointMLP(True);self.lstm=JointLSTM(True);self.scales=[]
  for model,file in [(self.mlp,'mlp5.pth'),(self.lstm,'lstm_full.pth')]:
   ck=torch.load(folder/file,map_location='cpu',weights_only=False)
   if ck.get('velocity_method')!='LinearFit5' or ck['velocity_scaling']!=self.spec['velocity_scaling']:raise ValueError('Model/velocity scaler mismatch')
   model.load_state_dict(ck['state_dict']);model.eval();self.scales.append((float(ck['mean_mm']),float(ck['std_mm'])))
  if samples>1:self.mlp.net[2].train()
  self.spline=BSpline(self.spec['knots'],np.asarray(self.spec['coefficients']).reshape(3,-1).T,self.spec['degree'],extrapolate=True)
  self.reset()
 def reset(self):
  self.history=deque(maxlen=5);self.initial=None;self.state=None;self.previous=None
 def gap(self):
  # Keep trial L0; discard temporal state after tracking loss or a frame gap.
  self.history.clear();self.state=None;self.previous=None
 @torch.inference_mode()
 def step(self,d,t,frame):
  if self.previous is not None and (frame-self.previous[1]!=1 or not 0<t-self.previous[0]<=.1):self.gap()
  if self.initial is None:self.initial=d
  self.previous=(t,frame);self.history.append((d,t))
  if len(self.history)<5:return {},{}
  distances,times=np.array(self.history).T;times=times-times[-1];times=times-times.mean();den=float(times@times)
  if den<=0:self.gap();return {},{}
  velocity=float(times@(distances-distances.mean())/den)
  scale=self.spec['velocity_scaling'];v=(velocity-scale['mean_mm_s'])/scale['std_mm_s'];history=distances/self.initial;x=float(history[-1]);outputs={};timings={}
  start=time.perf_counter();basis=self.spline((x-self.spec['train_x_min'])/self.spec['train_x_range']);outputs['function_mm']=float(basis@np.array([1.,self.size-3.,v]));timings['function_ms']=(time.perf_counter()-start)*1000
  start=time.perf_counter();features=torch.tensor([*history,self.size,v],dtype=torch.float32)
  if self.samples==1:result=self.mlp(features.unsqueeze(0)).item()
  else:result=torch.quantile(self.mlp(features.expand(self.samples,-1)),.5).item()
  mean,std=self.scales[0];outputs['mlp_mm']=result*std+mean;timings['mlp_ms']=(time.perf_counter()-start)*1000
  start=time.perf_counter();features=torch.tensor([[[x,self.size,v]]],dtype=torch.float32);hidden,self.state=self.lstm.lstm(features,self.state);result=self.lstm.head(hidden).item();mean,std=self.scales[1];outputs['lstm_mm']=result*std+mean;timings['lstm_ms']=(time.perf_counter()-start)*1000
  outputs.update(velocity_mm_s=velocity,distance_over_initial=x)
  return outputs,timings

def detect_marker(frame,settings,roi=None):
 hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV);mask=cv2.inRange(hsv,np.array(settings[0],np.uint8),np.array(settings[1],np.uint8))
 if roi:
  x,y,w,h=roi;region=np.zeros_like(mask);region[y:y+h,x:x+w]=255;mask=cv2.bitwise_and(mask,region)
 kernel=np.ones((3,3),np.uint8);mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel);mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
 contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);candidates=[]
 for c in contours:
  area=cv2.contourArea(c)
  if not 15<=area<=15000:continue
  m=cv2.moments(c)
  if m['m00']>0:candidates.append((area,np.array([m['m10']/m['m00'],m['m01']/m['m00']]),c))
 if not candidates:return None,None,mask
 _,center,contour=max(candidates,key=lambda a:a[0]);return center,contour,mask

def chart(rows):
 image=np.full((360,960,3),255,np.uint8);left,top,right,bottom=65,45,935,305
 xmax=max(3.,rows[-1]['time_s'] if rows else 3.);finite=[r[k] for r in rows for k in KEYS if np.isfinite(r[k])];ymax=max(125.,max(finite,default=0)*1.08)
 for y in np.linspace(0,ymax,6):
  py=int(bottom-y/ymax*(bottom-top));cv2.line(image,(left,py),(right,py),(230,230,230),1);cv2.putText(image,str(round(y)),(8,py+5),0,.45,(70,70,70),1)
 for t in np.linspace(0,xmax,7):
  px=int(left+t/xmax*(right-left));cv2.line(image,(px,top),(px,bottom),(235,235,235),1);cv2.putText(image,f'{t:.1f}',(px-12,bottom+20),0,.4,(70,70,70),1)
 for k,color,label,x in zip(KEYS,COLORS,['Function','MLP','LSTM'],[100,330,520]):
  cv2.line(image,(x,20),(x+30,20),color,2);cv2.putText(image,label,(x+40,25),0,.55,(30,30,30),1)
  previous=None
  for r in rows:
   y=r[k]
   if not np.isfinite(y):previous=None;continue
   pt=(int(left+r['time_s']/xmax*(right-left)),int(bottom-y/ymax*(bottom-top)))
   if previous:cv2.line(image,previous,pt,color,2,cv2.LINE_AA)
   previous=pt
 cv2.putText(image,'Time (s)',(450,350),0,.55,(40,40,40),1);cv2.putText(image,'Predicted aperture (mm)',(690,25),0,.5,(40,40,40),1)
 return image

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--participant',choices=['song','d1'],default='song');parser.add_argument('--models',type=Path);parser.add_argument('--size-cm',type=int,choices=[2,3,4],default=3)
 parser.add_argument('--camera',type=int,default=1);parser.add_argument('--settings',type=Path);parser.add_argument('--mm-per-px',type=float);parser.add_argument('--fps',type=float,default=30)
 parser.add_argument('--mlp-samples',type=int,default=1,help='1: standard inference, Dropout off; 128: MC Dropout median')
 parser.add_argument('--output',type=Path,default=ROOT/'Data/Marker_Realtime');parser.add_argument('--check',action='store_true')
 args=parser.parse_args()
 if args.mlp_samples<1 or args.fps<=0:parser.error('Invalid sample count/FPS')
 torch.set_num_threads(1);engine=Engine(args.models or BUNDLES[args.participant],args.size_cm,args.mlp_samples)
 data=ROOT/'Data'/('1115_Song' if args.participant=='song' else '1101_D');sizefolder=str(args.size_cm*10)+('' if args.participant=='song' else 'mm')
 settings_path=args.settings or next((data/sizefolder).glob('202*/settings.json'));settings=json.loads(settings_path.read_text(encoding='utf-8-sig'));scale=args.mm_per_px or float(settings.get('pixel_calibration',{}).get('mm_per_px',float('nan')))
 if not np.isfinite(scale) or scale<=0:parser.error('Valid mm/px calibration is required')
 if args.check:
  for i in range(30):pred,_=engine.step(300-i*3,100+i/30,i)
  assert all(np.isfinite(pred[k]) for k in KEYS)
  test=np.zeros((100,100,3),np.uint8);cv2.circle(test,(50,50),8,(0,180,0),-1);center,_,_=detect_marker(test,[(40,70,40),(95,255,255)]);assert center is not None and np.linalg.norm(center-[50,50])<1
  print('PASS: LinearFit5 model/scaler checks, finite predictions, synthetic green-marker localization; no camera opened.');return
 cap=cv2.VideoCapture(args.camera,cv2.CAP_DSHOW);cap.set(cv2.CAP_PROP_FRAME_WIDTH,settings.get('camera_source_width',640));cap.set(cv2.CAP_PROP_FRAME_HEIGHT,settings.get('camera_source_height',480));cap.set(cv2.CAP_PROP_FPS,args.fps)
 if not cap.isOpened():raise RuntimeError(f'Camera {args.camera} unavailable; try --camera 0')
 margins=settings.get('crop_margins_px',{});target=[None];roi=None;active=False;rows=[];session=None;writer=None;frame_id=0;t0=None;last=None
 cv2.namedWindow(WINDOW,cv2.WINDOW_NORMAL);cv2.resizeWindow(WINDOW,1100,850)
 def mouse(event,x,y,flags,param):
  if event==cv2.EVENT_LBUTTONDOWN and last is not None and 0<=x<last.shape[1] and 0<=y<last.shape[0]:
   if not active:target[0]=np.array([float(x),float(y)])
 cv2.setMouseCallback(WINDOW,mouse)
 cv2.namedWindow('Green mask',cv2.WINDOW_NORMAL)
 for label,value,maximum in [('H min',40,179),('H max',95,179),('S min',70,255),('V min',40,255)]:cv2.createTrackbar(label,'Green mask',value,maximum,lambda x:None)
 def stop():
  nonlocal active,writer
  if writer is not None:writer.release();writer=None
  if session and rows:
   with (session/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
   print('Saved',session,flush=True)
  active=False
 try:
  while True:
   began=time.perf_counter();ok,raw=cap.read();received=time.perf_counter();timestamp=time.time()
   if not ok:break
   h,w=raw.shape[:2]
   expected_w=int(settings.get('camera_source_width',w));expected_h=int(settings.get('camera_source_height',h))
   if args.mm_per_px is None and (w,h)!=(expected_w,expected_h):raise ValueError('Captured resolution differs from calibration; use matching camera resolution or recalibrate and provide --mm-per-px')
   l,t,r,b=[int(margins.get(k,0)) for k in ['left','top','right','bottom']]
   if l+r>=w or t+b>=h:raise ValueError('Calibration crop exceeds captured dimensions')
   frame=raw[t:h-b,l:w-r].copy();last=frame.copy();frame_id+=1
   thresholds=[(cv2.getTrackbarPos('H min','Green mask'),cv2.getTrackbarPos('S min','Green mask'),cv2.getTrackbarPos('V min','Green mask')),(cv2.getTrackbarPos('H max','Green mask'),255,255)]
   detection_start=time.perf_counter();marker,contour,mask=detect_marker(frame,thresholds,roi);detect_ms=(time.perf_counter()-detection_start)*1000
   values={k:float('nan') for k in KEYS};timing={k:float('nan') for k in ['function_ms','mlp_ms','lstm_ms']};distance=float('nan')
   if contour is not None:cv2.drawContours(frame,[contour],-1,(0,255,255),1);cv2.circle(frame,tuple(marker.astype(int)),4,(0,255,255),-1)
   if target[0] is not None:
    center=tuple(target[0].astype(int));cv2.drawMarker(frame,center,(255,180,40),cv2.MARKER_CROSS,20,2)
    if marker is not None:
     distance=float(np.linalg.norm(marker-target[0])*scale);cv2.line(frame,tuple(marker.astype(int)),center,(0,200,240),1)
     if active:
      output,measured=engine.step(distance,received,frame_id);values.update(output);timing.update(measured)
   if active and (marker is None or target[0] is None):engine.gap()
   if active:
    record=dict(frame=frame_id,time_s=received-t0,timestamp_unix_s=timestamp,marker_valid=int(marker is not None),marker_x=float(marker[0]) if marker is not None else float('nan'),marker_y=float(marker[1]) if marker is not None else float('nan'),distance_mm=distance,distance_over_initial=values.get('distance_over_initial',float('nan')),velocity_mm_s=values.get('velocity_mm_s',float('nan')),**{k:values[k] for k in KEYS},marker_detection_ms=detect_ms,**timing,processing_after_capture_ms=(time.perf_counter()-received)*1000)
    rows.append(record)
   if marker is not None and target[0] is not None and np.isfinite(values['lstm_mm']):
    direction=target[0]-marker;direction/=max(np.linalg.norm(direction),1e-6);normal=np.array([-direction[1],direction[0]]);palm=marker+direction*(25/scale);aperture=max(0,values['lstm_mm'])/scale
    tips=[palm+normal*aperture/2,palm-normal*aperture/2]
    for tip in tips:cv2.line(frame,tuple(marker.astype(int)),tuple(tip.astype(int)),COLORS[2],2);cv2.circle(frame,tuple(tip.astype(int)),5,COLORS[2],-1)
   put(frame,'GREEN MARKER -> PREDICTION (no hand measurement)',(8,20),scale=.43)
   put(frame,f'{"REC" if active else "READY"} | {args.size_cm*10} mm cube | Scale {scale:.3f} mm/px',(8,43))
   put(frame,f'Distance: {distance:.1f} mm',(8,65))
   for j,(key,label,color) in enumerate(zip(KEYS,['Function','MLP','LSTM'],COLORS)):put(frame,f'{label}: {values[key]:.1f} mm',(8,88+j*21),color)
   put(frame,'Green fingers: schematic aperture overlay',(8,frame.shape[0]-31),COLORS[2],.43)
   put(frame,'Click target | N start S stop | R marker ROI | C clear ROI | Q quit',(8,frame.shape[0]-10),scale=.4)
   graph=chart(rows);canvas=np.zeros((frame.shape[0]+360,max(frame.shape[1],960),3),np.uint8);canvas[:frame.shape[0],:frame.shape[1]]=frame;canvas[frame.shape[0]:,:960]=graph
   cv2.imshow(WINDOW,canvas);cv2.imshow('Green mask',mask)
   if writer is not None:writer.write(canvas)
   key=cv2.waitKey(1)&255
   if key in [ord('q'),27]:break
   if key==ord('s'):stop()
   if key==ord('c') and not active:roi=None
   if key==ord('r') and not active:
    chosen=cv2.selectROI('Select green marker search area',last,False,False);cv2.destroyWindow('Select green marker search area');roi=tuple(map(int,chosen)) if chosen[2]>0 and chosen[3]>0 else None
   if key==ord('n') and not active:
    if marker is None or target[0] is None:print('Click target center and make the green marker visible before N.',flush=True);continue
    engine.reset();rows=[];t0=time.perf_counter();session=args.output/f'marker_{datetime.now():%Y%m%d_%H%M%S_%f}';session.mkdir(parents=True)
    metadata=dict(models=str(engine.folder),participant_model=args.participant,size_cm=args.size_cm,velocity_method='LinearFit5',mlp_samples=args.mlp_samples,mlp_dropout_at_inference=args.mlp_samples>1,mm_per_px=scale,calibration_source=str(settings_path),camera=args.camera,camera_resolution=[w,h],crop_margins=margins,target_center_px=target[0].tolist(),marker_hsv=thresholds,marker_color='green',marker_roi=roi,overlay='schematic finger placement; only aperture is predicted, fingertip geometry is not learned',measurements='Marker center replaces index MCP; target center locked manually; no aperture ground truth measured',video_timing='Nominal FPS preview recording, not wall-clock synchronized playback. Use CSV timestamps for timing.',weights='existing fold_01 checkpoint by default, trained on 480 trials; not a full-600 fitted model')
    (session/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    writer=cv2.VideoWriter(str(session/'live_preview.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),args.fps,(canvas.shape[1],canvas.shape[0]))
    if not writer.isOpened():writer.release();writer=None;raise RuntimeError('Cannot initialize MP4 writer')
    active=True;print('Started',session,flush=True)
   elapsed=time.perf_counter()-began
   if elapsed<1/args.fps:time.sleep(1/args.fps-elapsed)
 finally:stop();cap.release();cv2.destroyAllWindows()
if __name__=='__main__':main()
