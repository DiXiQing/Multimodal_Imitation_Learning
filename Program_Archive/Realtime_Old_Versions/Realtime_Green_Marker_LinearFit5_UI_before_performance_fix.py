"""Green-marker live aperture estimation with the previous three-window control UI. N start, S stop, click target, Q quit.
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
from marker_hd_support import detect_cube,draw_gripper,hd_chart,compose,HDRecorder
from Train_D1_Joint_Three_Models_LinearFit5_CV5 import JointMLP,JointLSTM
ROOT=Path(__file__).resolve().parents[1]
BUNDLES={
 'song':ROOT/'Data/1115_Song/Result/Joint_Three_Models_LinearFit5_CurrentV_CV5_20261123_153753/fold_01',
 'd1':ROOT/'Data/1101_D_PreAction15/Result/Joint_Three_Models_LinearFit5_CurrentV_CV5_20261123_153838/fold_01'}
KEYS=['function_mm','mlp_mm','lstm_mm']
COLORS=[(190,125,45),(40,155,245),(75,170,60)]
WINDOW='Green marker | Auto cube detection | N start S stop Q quit'

def put(im,text,xy,color=(235,235,235),scale=.5):
 factor=max(1.,im.shape[1]/800)
 xy=(round(xy[0]*factor),round(xy[1]*factor) if xy[1]<200 else im.shape[0]-round((im.shape[0]-xy[1])*factor))
 scale*=factor
 cv2.putText(im,text,xy,cv2.FONT_HERSHEY_SIMPLEX,scale,(15,15,15),max(3,round(factor*2)),cv2.LINE_AA)
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
 parser.add_argument('--width',type=int,default=1920);parser.add_argument('--height',type=int,default=1080)
 parser.add_argument('--crop-left',type=int,default=150);parser.add_argument('--crop-right',type=int,default=100);parser.add_argument('--crop-bottom',type=int,default=50)
 parser.add_argument('--object-color',choices=['black','blue','red'],default=None)
 parser.add_argument('--finger-length-mm',type=float,default=30)
 parser.add_argument('--mlp-samples',type=int,default=1,help='1: standard inference, Dropout off; 128: MC Dropout median')
 parser.add_argument('--output',type=Path,default=ROOT/'Data/Marker_Realtime');parser.add_argument('--check',action='store_true')
 args=parser.parse_args()
 if min(args.crop_left,args.crop_right,args.crop_bottom)<0:parser.error('Crop margins must be nonnegative')
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
 cap=cv2.VideoCapture(args.camera,cv2.CAP_DSHOW);cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'));cap.set(cv2.CAP_PROP_FRAME_WIDTH,args.width);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,args.height);cap.set(cv2.CAP_PROP_FPS,args.fps)
 if not cap.isOpened():raise RuntimeError(f'Camera {args.camera} unavailable; try --camera 0')
 margins=dict(settings.get('crop_margins_px',{}))
 # Additional native-camera pixels removed from the previously displayed crop.
 margins['left']=int(margins.get('left',0))+args.crop_left
 margins['right']=int(margins.get('right',0))+args.crop_right
 margins['bottom']=int(margins.get('bottom',0))+args.crop_bottom
 target=[None];roi=None;active=False;rows=[];session=None;writer=None;frame_id=0;t0=None;last=None
 import tkinter as tk
 from tkinter import messagebox,simpledialog
 ui=tk.Tk();ui.title('三模型实时对比 — 绿色 Marker');ui.geometry('460x980')
 status=tk.StringVar(value='自动检测方块与绿色 Marker。高清模式首次使用按 K 做两点标定，再按 N 开始。')
 calibration_ready=[args.mm_per_px is not None];calibration_points=[];calibration_length=[None];manual_target=[False];calibration_info=[None]
 selected_size=tk.IntVar(value=args.size_cm)
 distance_label=tk.StringVar(value='Marker–物体距离 — mm')
 tk.Label(ui,text='绿色 Marker | LinearFit5',font=('',15,'bold')).pack(pady=8)
 controls=[]
 size_row=tk.Frame(ui);size_row.pack()
 def switch_size():
  if active:
   selected_size.set(engine.size);return
  args.size_cm=selected_size.get();engine.size=args.size_cm;engine.reset()
  status.set(f'已切换为 {args.size_cm*10} mm，开始前确认方块中心和标定。')
 for size in [2,3,4]:
  button=tk.Radiobutton(size_row,text=f'{size*10} mm',variable=selected_size,value=size,command=switch_size);button.pack(side='left',padx=8);controls.append(button)
 tk.Label(ui,textvariable=distance_label,font=('',16,'bold')).pack(pady=8)
 calibration_text=tk.StringVar(value=f'Calibration: {scale:.6g} mm/px (checking resolution)');tk.Label(ui,textvariable=calibration_text).pack()
 camera_info=tk.StringVar(value='Camera resolution: opening');tk.Label(ui,textvariable=camera_info).pack()
 tk.Label(ui,text=f'{args.participant.upper()} | MLP samples: {args.mlp_samples}',font=('',10)).pack()
 tk.Label(ui,textvariable=status,wraplength=400,justify='left',fg='#167c86').pack(fill='x',padx=12,pady=8)
 target_distance=tk.IntVar(value=300);target_duration=tk.IntVar(value=2)
 row=tk.Frame(ui);row.pack()
 for value in [250,300,350]:
  button=tk.Radiobutton(row,text=f'{value} mm 起始条件',variable=target_distance,value=value);button.pack(side='left');controls.append(button)
 row=tk.Frame(ui);row.pack()
 for value in [1,2,3]:
  button=tk.Radiobutton(row,text=f'{value} 秒条件',variable=target_duration,value=value);button.pack(side='left',padx=15);controls.append(button)
 hsv_vars={name:tk.IntVar(value=value) for name,value in [('H min',40),('H max',95),('S min',70),('V min',40)]}
 for name,label in [('H min','绿色色相下限 H'),('H max','绿色色相上限 H'),('S min','绿色饱和度下限 S'),('V min','绿色亮度下限 V')]:
  slider=tk.Scale(ui,label=label,variable=hsv_vars[name],from_=0,to=179 if name.startswith('H') else 255,orient='horizontal',length=350);slider.pack();controls.append(slider)
 object_color=tk.StringVar(value=args.object_color or settings.get('object_color','black'))
 row=tk.Frame(ui);row.pack();tk.Label(row,text='方块颜色').pack(side='left')
 for label,color in [('黑色','black'),('蓝色','blue'),('红色','red')]:
  button=tk.Radiobutton(row,text=label,variable=object_color,value=color);button.pack(side='left');controls.append(button)
 cube_v=tk.IntVar(value=60);cube_min=tk.IntVar(value=100)
 for label,var,maximum in [('黑色亮度上限',cube_v,255),('方块最小面积',cube_min,10000)]:
  slider=tk.Scale(ui,label=label,variable=var,from_=0,to=maximum,orient='horizontal',length=350);slider.pack();controls.append(slider)
 show_mask=tk.BooleanVar(value=False);tk.Checkbutton(ui,text='显示绿色检测掩膜',variable=show_mask).pack()
 estimate_vars=[tk.StringVar(value=f'{label}: — mm') for label in ['Function','MLP','LSTM']]
 for var in estimate_vars:tk.Label(ui,textvariable=var,font=('',11)).pack()
 pending=[];closing=[False]
 def request(key):pending.append(ord(key))
 for label,key in [('开始一次 (N)','n'),('结束并保存 (S)','s'),('自动检测 / 手动点选中心 (L)','l'),('两点距离标定 (K)','k'),('框选 Marker 搜索区域 (R)','r'),('清除搜索区域 (C)','c')]:
  tk.Button(ui,text=label,command=lambda k=key:request(k)).pack(fill='x',padx=20,pady=2)
 tk.Button(ui,text='退出 (Q)',command=lambda:closing.__setitem__(0,True)).pack(fill='x',padx=20,pady=2)
 ui.protocol('WM_DELETE_WINDOW',lambda:closing.__setitem__(0,True))
 def shortcut(event):
  if event.char.lower() in 'nslrckq' and event.char:request(event.char.lower())
 ui.bind('<Key>',shortcut)
 cv2.namedWindow(WINDOW,cv2.WINDOW_NORMAL);cv2.resizeWindow(WINDOW,700,500)
 cv2.namedWindow('Live aperture comparison',cv2.WINDOW_NORMAL);cv2.resizeWindow('Live aperture comparison',960,360)
 def mouse(event,x,y,flags,param):
  nonlocal scale
  if event==cv2.EVENT_LBUTTONDOWN and last is not None and 0<=x<last.shape[1] and 0<=y<last.shape[0] and not active:
   point=np.array([float(x),float(y)])
   if calibration_length[0] is not None:
    calibration_points.append(point)
    if len(calibration_points)==2:
     pixels=float(np.linalg.norm(calibration_points[1]-calibration_points[0]))
     if pixels<5:status.set('两点过近，请重新标定。')
     else:
      scale=calibration_length[0]/pixels;calibration_ready[0]=True;engine.reset();calibration_text.set(f'Calibration: {scale:.6g} mm/px');calibration_info[0]=dict(length_mm=calibration_length[0],points=[p.tolist() for p in calibration_points]);status.set('标定完成。确认检测后按 N 开始。')
     calibration_points.clear();calibration_length[0]=None
   elif manual_target[0]:target[0]=point;status.set('已手动锁定物体中心；按 L 恢复自动检测。')
 cv2.setMouseCallback(WINDOW,mouse)
 def stop():
  nonlocal active,writer
  if writer is not None:writer.release();writer=None
  if session and rows:
   with (session/'predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
   print('Saved',session,flush=True)
  active=False
  for control in controls:control.config(state='normal')
  if session and rows:status.set(f'已保存高清同步视频和 CSV：{session.name}')
 try:
  while not closing[0]:
   ui.update_idletasks();ui.update()
   began=time.perf_counter();ok,raw=cap.read();received=time.perf_counter();timestamp=time.time()
   if not ok:break
   h,w=raw.shape[:2]
   expected_w=int(settings.get('camera_source_width',w));expected_h=int(settings.get('camera_source_height',h))
   if args.mm_per_px is None and (w,h)!=(expected_w,expected_h) and calibration_info[0] is None:calibration_ready[0]=False
   if args.mm_per_px is None and (w,h)==(expected_w,expected_h):calibration_ready[0]=True
   camera_info.set(f'Camera: {w} × {h} | Output: 1920 × 1080')
   if not calibration_ready[0]:calibration_text.set('需要标定：按 K，输入已知长度并点两端')
   l,t,r,b=[int(margins.get(k,0)) for k in ['left','top','right','bottom']]
   if l+r>=w or t+b>=h:raise ValueError('Calibration crop exceeds captured dimensions')
   frame=raw[t:h-b,l:w-r].copy();last=frame.copy();frame_id+=1
   thresholds=[(hsv_vars['H min'].get(),hsv_vars['S min'].get(),hsv_vars['V min'].get()),(hsv_vars['H max'].get(),255,255)]
   detection_start=time.perf_counter();marker,contour,mask=detect_marker(frame,thresholds,roi)
   (cube_area,cube_box),cube_mask=detect_cube(frame,black_max=cube_v.get(),saturation_max=120,min_area=max(1,cube_min.get()),color=object_color.get(),color_s_min=100,color_v_min=45,max_area=frame.shape[0]*frame.shape[1]*.3)
   if not active and not manual_target[0]:target[0]=np.array([cube_box[0]+cube_box[2]/2,cube_box[1]+cube_box[3]/2]) if cube_box else None
   if cube_box is not None:
    bx,by,bw,bh=cube_box;cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),(30,180,255),2)
   detect_ms=(time.perf_counter()-detection_start)*1000
   values={k:float('nan') for k in KEYS};timing={k:float('nan') for k in ['function_ms','mlp_ms','lstm_ms']};distance=float('nan')
   if contour is not None:cv2.drawContours(frame,[contour],-1,(0,255,255),1);cv2.circle(frame,tuple(marker.astype(int)),4,(0,255,255),-1)
   if target[0] is not None:
    center=tuple(target[0].astype(int));cv2.drawMarker(frame,center,(255,180,40),cv2.MARKER_CROSS,20,2)
    if marker is not None:
     distance=float(np.linalg.norm(marker-target[0])*scale) if calibration_ready[0] else float('nan')
     if active and np.isfinite(distance):
      output,measured=engine.step(distance,received,frame_id);values.update(output);timing.update(measured)
   if active and (marker is None or target[0] is None):engine.gap()
   if active:
    record=dict(frame=frame_id,time_s=received-t0,timestamp_unix_s=timestamp,marker_valid=int(marker is not None),marker_x=float(marker[0]) if marker is not None else float('nan'),marker_y=float(marker[1]) if marker is not None else float('nan'),distance_mm=distance,distance_over_initial=values.get('distance_over_initial',float('nan')),velocity_mm_s=values.get('velocity_mm_s',float('nan')),**{k:values[k] for k in KEYS},marker_detection_ms=detect_ms,**timing,processing_after_capture_ms=(time.perf_counter()-received)*1000)
    rows.append(record)
   draw_gripper(frame,marker,target[0],values['lstm_mm'],scale if calibration_ready[0] else float('nan'),args.finger_length_mm)
   put(frame,'GREEN MARKER -> APERTURE PREDICTION',(8,20),scale=.43)
   put(frame,f'{"REC" if active else "READY"} | {args.size_cm*10} mm cube | Scale {scale:.3f} mm/px',(8,43))
   put(frame,f'Distance: {distance:.1f} mm',(8,65))
   for j,(key,label,color) in enumerate(zip(KEYS,['Function','MLP','LSTM'],COLORS)):put(frame,f'{label}: {values[key]:.1f} mm',(8,88+j*21),color)
   put(frame,'Teal jaws / pink tips: schematic LSTM aperture overlay',(8,frame.shape[0]-31),COLORS[2],.43)
   put(frame,'Auto target | K calibrate | N start S save | L manual | Q quit',(8,frame.shape[0]-10),scale=.4)
   graph=hd_chart(rows);canvas=compose(frame,graph)
   cv2.imshow(WINDOW,frame);cv2.imshow('Live aperture comparison',graph)
   distance_label.set(f'Marker–物体距离 {distance:.1f} mm' if np.isfinite(distance) else 'Marker–物体距离 — mm')
   for var,key in zip(estimate_vars,KEYS):var.set(f'{key.split("_")[0].upper()}: {values[key]:.1f} mm')
   if show_mask.get():cv2.imshow('Green mask',mask)
   else:
    try:cv2.destroyWindow('Green mask')
    except cv2.error:pass
   if writer is not None:writer.write(canvas,frame,received-t0)
   key=cv2.waitKey(1)&255
   if pending:key=pending.pop(0)
   if key in [ord('q'),27]:break
   if key==ord('s'):stop()
   if key==ord('l') and not active:
    manual_target[0]=not manual_target[0];target[0]=None;engine.reset();status.set('请点击物体中心。' if manual_target[0] else '已恢复自动方块检测。')
   if key==ord('k') and not active:
    length=simpledialog.askfloat('两点距离标定','请输入参考物两端的真实长度（mm），再在相机画面点两端：',initialvalue=100,minvalue=.1,parent=ui)
    if length is not None:calibration_length[0]=length;calibration_points.clear();status.set('请在相机画面点击已知长度的两个端点。')
   if key==ord('c') and not active:roi=None
   if key==ord('r') and not active:
    chosen=cv2.selectROI('Select green marker search area',last,False,False);cv2.destroyWindow('Select green marker search area');roi=tuple(map(int,chosen)) if chosen[2]>0 and chosen[3]>0 else None
   if key==ord('n') and not active:
    if not calibration_ready[0]:status.set('请先按 K 完成当前分辨率的距离标定。');continue
    if marker is None or target[0] is None:status.set('需要同时检测到绿色 Marker 和方块；可调整方块颜色和亮度阈值。');continue
    engine.reset();rows=[];t0=time.perf_counter();session=args.output/f'marker_{datetime.now():%Y%m%d_%H%M%S_%f}';session.mkdir(parents=True)
    metadata=dict(models=str(engine.folder),participant_model=args.participant,size_cm=args.size_cm,velocity_method='LinearFit5',mlp_samples=args.mlp_samples,mlp_dropout_at_inference=args.mlp_samples>1,mm_per_px=scale,calibration_source=str(settings_path),camera=args.camera,camera_resolution=[w,h],crop_margins=margins,target_center_px=target[0].tolist(),marker_hsv=thresholds,marker_color='green',marker_roi=roi,target_distance_mm=target_distance.get(),target_movement_s=target_duration.get(),overlay='schematic finger placement; only aperture is predicted, fingertip geometry is not learned',measurements='Marker center replaces index MCP; target auto-detected and locked at recording start, or manually overridden; no aperture ground truth measured',video_timing='Timestamp-resampled constant FPS using preceding captured frame; output 1920x1080 MJPEG AVI and native-resolution camera overlay AVI',weights='existing fold_01 checkpoint by default, trained on 480 trials; not a full-600 fitted model')
    (session/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    metadata.update(object_detection='automatic HSV largest filled contour',object_color=object_color.get(),object_center_manual=manual_target[0],calibration_two_points=calibration_info[0],capture_requested=[args.width,args.height],native_cropped_size=[frame.shape[1],frame.shape[0]],export_size=[1920,1080],export_codec='MJPEG AVI',finger_length_mm=args.finger_length_mm)
    (session/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    writer=HDRecorder(session,args.fps,(frame.shape[1],frame.shape[0]))
    active=True
    for control in controls:control.config(state='disabled')
    status.set('录制预测中；按 S 结束并保存。')
    print('Started',session,flush=True)
   elapsed=time.perf_counter()-began
   if elapsed<1/args.fps:time.sleep(1/args.fps-elapsed)
 finally:
  stop();cap.release();cv2.destroyAllWindows()
  try:ui.destroy()
  except tk.TclError:pass
if __name__=='__main__':main()
