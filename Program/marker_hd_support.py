"""Rendering, automatic cube detection and timestamp-aligned recording helpers."""
from pathlib import Path
import cv2,numpy as np
from Realtime_Joint_Three_Models_LinearFit7 import detect_cube
ORANGE=(46,124,215);TEAL=(139,130,23);PINK=(144,73,205)
def draw_gripper(frame,marker,target,aperture_mm,mm_per_px,reach_mm=30):
 if marker is None or target is None:return
 a=np.asarray(marker,float);b=np.asarray(target,float);delta=b-a;direction=delta/max(float(np.linalg.norm(delta)),1e-6);normal=np.array([-direction[1],direction[0]])
 def pt(x):return tuple(np.rint(x).astype(int))
 thickness=max(2,round(frame.shape[1]/500));cv2.line(frame,pt(a),pt(b),ORANGE,max(2,thickness//2),cv2.LINE_AA);cv2.circle(frame,pt(b),3,(250,250,250),-1,cv2.LINE_AA)
 cv2.circle(frame,pt(a),max(6,thickness+2),ORANGE,-1,cv2.LINE_AA)
 if not np.isfinite(aperture_mm) or not np.isfinite(mm_per_px):return
 middle=a+direction*reach_mm/mm_per_px;opening=max(0,float(aperture_mm))/mm_per_px/2
 tips=[middle+normal*opening,middle-normal*opening]
 for tip in tips:
  cv2.line(frame,pt(a),pt(tip),(245,245,245),thickness+2,cv2.LINE_AA)
  cv2.line(frame,pt(a),pt(tip),TEAL,thickness,cv2.LINE_AA)
 cv2.line(frame,pt(tips[0]),pt(tips[1]),PINK,max(2,thickness//2),cv2.LINE_AA)
 for tip in tips:cv2.circle(frame,pt(tip),max(4,thickness),PINK,-1,cv2.LINE_AA)

def hd_chart(rows,width=1920,height=360):
 image=np.full((height,width,3),255,np.uint8);l,t,r,b=110,58,width-50,height-60
 xmax=max(3.,rows[-1]['time_s'] if rows else 3.);values=[row[k] for row in rows for k in ['function_mm','mlp_mm','lstm_mm'] if np.isfinite(row[k])];ymax=max(125.,max(values,default=0)*1.08)
 for value in np.linspace(0,ymax,6):
  py=round(b-value/ymax*(b-t));cv2.line(image,(l,py),(r,py),(225,225,225),1);cv2.putText(image,f'{value:.0f}',(28,py+7),0,.6,(70,70,70),1,cv2.LINE_AA)
 for value in np.linspace(0,xmax,7):
  px=round(l+value/xmax*(r-l));cv2.line(image,(px,t),(px,b),(235,235,235),1);cv2.putText(image,f'{value:.1f}',(px-16,b+27),0,.6,(70,70,70),1,cv2.LINE_AA)
 for key,label,color,x in [('function_mm','Function',(190,125,45),120),('mlp_mm','MLP',(40,155,245),420),('lstm_mm','LSTM',(75,170,60),680)]:
  cv2.line(image,(x,23),(x+45,23),color,3);cv2.putText(image,label,(x+55,31),0,.8,(40,40,40),1,cv2.LINE_AA);previous=None
  for row in rows:
   if not np.isfinite(row[key]):previous=None;continue
   point=(round(l+row['time_s']/xmax*(r-l)),round(b-row[key]/ymax*(b-t)))
   if previous is not None:cv2.line(image,previous,point,color,2,cv2.LINE_AA)
   previous=point
 cv2.putText(image,'Predicted aperture (mm)',(width-470,31),0,.7,(40,40,40),1,cv2.LINE_AA);cv2.putText(image,'Time (s)',(width//2-50,height-12),0,.65,(40,40,40),1,cv2.LINE_AA)
 return image

def compose(frame,graph):
 # Never stretch the camera aspect ratio; chart is drawn at native output size.
 canvas=np.full((1080,1920,3),246,np.uint8);ratio=min(1920/frame.shape[1],720/frame.shape[0]);size=(round(frame.shape[1]*ratio),round(frame.shape[0]*ratio));resized=cv2.resize(frame,size,interpolation=cv2.INTER_AREA if ratio<1 else cv2.INTER_CUBIC);x=(1920-size[0])//2;y=(720-size[1])//2;canvas[y:y+size[1],x:x+size[0]]=resized;canvas[720:]=graph;return canvas

class HDRecorder:
 def __init__(self,folder,fps,native_size,raw_size=None):
  self.folder=Path(folder);self.fps=fps;self.step=1/fps;self.next=0.;self.previous=None;self.count=0;self.frame_map=[];self.closed=False
  self.combined=cv2.VideoWriter(str(self.folder/'synchronized_demo_hd.avi'),cv2.VideoWriter_fourcc(*'MJPG'),fps,(1920,1080))
  self.camera=cv2.VideoWriter(str(self.folder/'camera_overlay_native.avi'),cv2.VideoWriter_fourcc(*'MJPG'),fps,native_size)
  self.curve=cv2.VideoWriter(str(self.folder/'prediction_curves.avi'),cv2.VideoWriter_fourcc(*'MJPG'),fps,(1920,360))
  self.raw=cv2.VideoWriter(str(self.folder/'camera_raw.avi'),cv2.VideoWriter_fourcc(*'MJPG'),fps,raw_size) if raw_size is not None else None
  writers=[self.combined,self.camera,self.curve]+([self.raw] if self.raw is not None else [])
  if not all(writer.isOpened() for writer in writers):self.release();raise RuntimeError('Cannot open MJPEG recording')
  for writer in writers:writer.set(cv2.VIDEOWRITER_PROP_QUALITY,95)
 def _emit(self,item):
  canvas,frame,raw,captured_t,source_frame_id=item
  self.combined.write(canvas);self.camera.write(frame);self.curve.write(canvas[720:1080])
  if self.raw is not None:
   if raw is None:raise ValueError('Raw recording requires original camera frames')
   self.raw.write(raw)
  self.frame_map.append(dict(video_frame=self.count,video_time_s=self.next,source_capture_time_s=captured_t,source_frame=source_frame_id))
  self.count+=1;self.next+=self.step
 def write(self,canvas,frame,t,raw=None,source_frame_id=None):
  # All outputs share one timestamp-resampled timeline, including duplicate frames.
  item=(canvas.copy(),frame.copy(),raw.copy() if raw is not None else None,float(t),source_frame_id)
  if self.previous is None:self.previous=item
  while self.next<t-1e-6:self._emit(self.previous)
  self.previous=item
  if abs(self.next-t)<=1e-6:self._emit(item)
 def release(self):
  if self.closed:return
  self.closed=True
  try:
   if self.previous is not None:self._emit(self.previous);self.previous=None
  finally:
   for writer in [getattr(self,'combined',None),getattr(self,'camera',None),getattr(self,'curve',None),getattr(self,'raw',None)]:
    if writer is not None:writer.release()
  if self.frame_map:
   import csv
   with (self.folder/'video_frame_timestamps.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(self.frame_map[0]));writer.writeheader();writer.writerows(self.frame_map)


class AsyncHDRecorder:
 """Bounded encoding worker: acquisition/UI never wait for a slow encoder."""
 def __init__(self,folder,fps,native_size,raw_size=None):
  import queue,threading
  self.recorder=HDRecorder(folder,fps,native_size,raw_size)
  self.queue=queue.Queue(maxsize=6);self.skipped=0;self.error=None
  self.thread=threading.Thread(target=self._work,daemon=True);self.thread.start()
 def _work(self):
  try:
   while True:
    item=self.queue.get()
    try:
     if item is None:break
     self.recorder.write(*item)
    finally:self.queue.task_done()
  except Exception as exc:self.error=exc
  finally:self.recorder.release()
 def write(self,canvas,frame,t,raw=None,source_frame_id=None):
  import queue
  if self.error is not None:raise RuntimeError('HD encoding failed') from self.error
  # Callers allocate new images each frame; ownership stays with this queue.
  try:self.queue.put_nowait((canvas,frame,t,raw,source_frame_id))
  except queue.Full:self.skipped+=1
 def release(self):
  import queue
  if not self.thread.is_alive():
   if self.error is not None:raise RuntimeError('HD encoding failed') from self.error
   return
  while self.thread.is_alive():
   try:self.queue.put(None,timeout=.1);break
   except queue.Full:continue
  self.thread.join()
  if self.error is not None:raise RuntimeError('HD encoding failed') from self.error
