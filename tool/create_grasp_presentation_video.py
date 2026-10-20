"""Create a presentation overlay from recorded coordinates and replayed MediaPipe skeleton."""
import sys,csv,json,subprocess
from pathlib import Path
import cv2,numpy as np,mediapipe as mp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'New/Program'))
from record_topview_black_cube import detect_cube,detect_fingers
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'New/Data/D1/3/Train/Blue_3cm_20261017_184913'
out=source/'presentation_overlay';out.mkdir(exist_ok=True)
rows=list(csv.DictReader((source/'camera.csv').open(encoding='utf-8-sig')));settings=json.loads((source/'settings.json').read_text())
cap=cv2.VideoCapture(str(source/'grasp_video.mp4'));fps=cap.get(cv2.CAP_PROP_FPS);w=int(cap.get(3));h=int(cap.get(4));assert int(cap.get(7))==len(rows)
writer=cv2.VideoWriter(str(out/'annotated_preview.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h+86));assert writer.isOpened()
hands=mp.solutions.hands.Hands(static_image_mode=False,max_num_hands=1,model_complexity=1,min_detection_confidence=.6,min_tracking_confidence=.6)
frames=[];skeletons=0
for i,r in enumerate(rows):
 ok,raw=cap.read();assert ok
 _,_,landmarks=detect_fingers(raw,hands)
 display=raw.copy()
 if landmarks:
  mp.solutions.drawing_utils.draw_landmarks(display,landmarks,mp.solutions.hands.HAND_CONNECTIONS);skeletons+=1
 (_,box),_=detect_cube(raw,settings['v'],settings['s'],settings['area'],color='blue',color_s_min=settings['color_s_min'],color_v_min=settings['color_v_min'])
 if box:
  x,y,bw,bh=box;cv2.rectangle(display,(x,y),(x+bw,y+bh),(0,255,0),2)
 def point(a,b):return (int(round(float(r[a]))),int(round(float(r[b])))) if r[a] and r[b] else None
 a=point('object_x','object_y');c=point('hand_center_x','hand_center_y');thumb=point('thumb_x','thumb_y');index=point('index_x','index_y')
 if a and c:cv2.line(display,a,c,(0,255,255),2)
 if thumb and index:cv2.line(display,thumb,index,(255,200,0),2)
 for pt,color,label in [(a,(0,0,255),''),(c,(0,255,255),''),(thumb,(255,200,0),''),(index,(255,200,0),'')]:
  if pt:
   cv2.circle(display,pt,4,color,-1)
   if label:cv2.putText(display,label,(pt[0]+7,pt[1]-7),0,.5,color,2,cv2.LINE_AA)
 canvas=np.full((h+86,w,3),(30,30,30),np.uint8);canvas[:h]=display
 lines=[('Blue cube | 30 mm',f'Frame {i+1:02d}/{len(rows)}  |  {float(r["segment_elapsed_s"]):.2f} s'),(f'Aperture B: {float(r["finger_width"]):.1f} px',f'Distance: {float(r["hand_object_distance"]):.1f} px')]
 for j,(left,right) in enumerate(lines):
  y=h+27+j*34;cv2.putText(canvas,left,(12,y),0,.55,(240,240,240),1,cv2.LINE_AA);cv2.putText(canvas,right,(330,y),0,.55,(240,240,240),1,cv2.LINE_AA)
 writer.write(canvas)
 if i in [0,22,35]:frames.append(canvas)
writer.release();cap.release();hands.close()
final=out/'Blue_3cm_20261017_184913_annotated.mp4'
try:
 import imageio_ffmpeg
 ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
 subprocess.run([ffmpeg,'-y','-i',str(out/'annotated_preview.mp4'),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(final)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
 codec='H264'
except ImportError:
 (out/'annotated_preview.mp4').replace(final);codec='mp4v'
check=cv2.VideoCapture(str(final));assert int(check.get(7))==len(rows);assert int(check.get(3))==w and int(check.get(4))==h+86;check.release()
cv2.imwrite(str(out/'preview.jpg'),np.hstack(frames))
(out/'method.json').write_text(json.dumps(dict(source=str(source),frames=len(rows),fps=fps,codec=codec,skeleton_frames=skeletons,method='Recorded CSV coordinates and metrics; MediaPipe skeleton replay with recorder parameters; object bounding box replayed with original HSV settings. Original sources unchanged.'),indent=2))
print(final);print('frames',len(rows),'skeletons',skeletons,'codec',codec)
