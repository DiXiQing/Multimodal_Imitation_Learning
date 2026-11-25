"""Rebuild trial 7 from raw video and recorded predictions; no inference."""
import argparse,csv,json
from pathlib import Path
import cv2
import numpy as np
from marker_hd_support import draw_gripper
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--trial',type=Path,default=ROOT/'Data/Marker_Realtime/marker_20261124_182225_575017');p.add_argument('--crop-y',type=int,default=215);p.add_argument('--crop-height',type=int,default=180);a=p.parse_args()
 folder=a.trial;out=folder/'Presentation_Strips';out.mkdir(exist_ok=True)
 meta=json.loads((folder/'metadata.json').read_text(encoding='utf-8-sig'))
 with (folder/'predictions.csv').open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
 with (folder/'video_frame_timestamps.csv').open(encoding='utf-8-sig') as f: mapping=list(csv.DictReader(f))
 byframe={int(r['frame']):r for r in rows};times=np.array([float(r['time_s']) for r in rows]);keys=['function_mm','mlp_mm','lstm_mm'];colors=[(190,125,45),(40,155,245),(75,170,60)]
 cap=cv2.VideoCapture(str(folder/'camera_raw.avi'));fps=cap.get(cv2.CAP_PROP_FPS)
 # Crop full horizontal travel; raw coordinates = stored cropped coordinates + actual margins.
 left=int(meta['actual_crop_margins']['left']);target=np.array(meta['target_center_px'])+[left,int(meta['actual_crop_margins']['top'])]
 width=1920;vh=round(a.crop_height*width/(640-left));ch=440;vh+=vh%2
 writer=cv2.VideoWriter(str(out/'trial07_stacked_strips.avi'),cv2.VideoWriter_fourcc(*'MJPG'),fps,(width,vh+ch))
 if not writer.isOpened():raise RuntimeError('Cannot create output video')
 xmax=np.ceil(max(times)*2)/2
 l,r,t,b=145,1875,62,365
 points={k:[] for k in keys}
 for row in rows:
  for k in keys:
   v=float(row[k]);points[k].append((round(l+float(row['time_s'])/xmax*(r-l)),round(b-v/125*(b-t))) if np.isfinite(v) else None)
 def chart(now):
  xmax=max(1.5,now)
  im=np.full((ch,width,3),255,np.uint8)
  for val in range(0,126,25):
   y=round(b-val/125*(b-t));cv2.line(im,(l,y),(r,y),(230,230,230),1,cv2.LINE_AA);cv2.putText(im,str(val),(85,y+8),0,.75,(55,55,55),1,cv2.LINE_AA)
  for val in np.arange(0,xmax+.01,.5):
   x=round(l+val/xmax*(r-l));cv2.line(im,(x,t),(x,b),(235,235,235),1);cv2.putText(im,f'{val:.1f}',(x-18,b+32),0,.75,(55,55,55),1,cv2.LINE_AA)
  cv2.line(im,(l,t),(l,b),(70,70,70),1);cv2.line(im,(l,b),(r,b),(70,70,70),1)
  for k,label,color,x in zip(keys,['Function','MLP','LSTM'],colors,[150,470,750]):
   cv2.line(im,(x,25),(x+50,25),color,4,cv2.LINE_AA);cv2.putText(im,label,(x+65,35),0,.9,(35,35,35),2,cv2.LINE_AA)
   prev=None
   for j in range(np.searchsorted(times,now,side='right')):
    v=float(rows[j][k])
    pt=(round(l+times[j]/xmax*(r-l)),round(b-v/125*(b-t))) if np.isfinite(v) else None
    if prev is not None and pt is not None:cv2.line(im,prev,pt,color,3,cv2.LINE_AA)
    prev=pt
  text='Predicted aperture (mm)';font=cv2.FONT_HERSHEY_SIMPLEX;fs=.85
  (tw,th),base=cv2.getTextSize(text,font,fs,1)
  label=np.full((th+base+10,tw+10,3),255,np.uint8)
  cv2.putText(label,text,(5,5+th),font,fs,(40,40,40),1,cv2.LINE_AA)
  label=cv2.rotate(label,cv2.ROTATE_90_COUNTERCLOCKWISE)
  ly=round((t+b-label.shape[0])/2);lx=round(38-label.shape[1]/2)
  im[ly:ly+label.shape[0],lx:lx+label.shape[1]]=label
  text='Time (s)';(tw,th),base=cv2.getTextSize(text,font,fs,1)
  cv2.putText(im,text,(round((l+r-tw)/2),425),font,fs,(40,40,40),1,cv2.LINE_AA)
  return im
 count=0
 try:
  for m in mapping:
   ok,raw=cap.read()
   if not ok:raise RuntimeError('Raw video shorter than timestamp map')
   row=byframe.get(int(m['source_frame']))
   if row is None:raise RuntimeError('Missing recorded prediction')
   crop=raw[a.crop_y:a.crop_y+a.crop_height,left:]
   if crop.shape[:2]!=(a.crop_height,640-left):raise ValueError('Invalid crop')
   bright=float(row.get('display_brightness',15));contrast=float(row.get('display_contrast_percent',105))/100
   crop=np.clip(crop.astype(float)*contrast+bright,0,255).astype(np.uint8);view=cv2.resize(crop,(width,vh),interpolation=cv2.INTER_CUBIC);scale=width/crop.shape[1]
   if int(row['marker_valid']):
    marker=(np.array([float(row['marker_x'])+left,float(row['marker_y'])+meta['actual_crop_margins']['top']])-[left,a.crop_y])*scale
    tar=(target-[left,a.crop_y])*scale
    draw_gripper(view,marker,tar,float(row['lstm_mm']),meta['mm_per_px']/scale,meta.get('finger_length_mm',30))
    aperture_value=float(row['lstm_mm'])
    if np.isfinite(aperture_value):
     direction=tar-marker;direction/=max(float(np.linalg.norm(direction)),1e-6)
     normal=np.array([-direction[1],direction[0]])
     middle=marker+direction*meta.get('finger_length_mm',30)/meta['mm_per_px']*scale
     opening=max(0,aperture_value)/meta['mm_per_px']*scale/2
     tips=[middle+normal*opening,middle-normal*opening]
     pt=lambda q:tuple(np.rint(q).astype(int))
     thick=8
     for tip in tips:
      cv2.line(view,pt(marker),pt(tip),(245,245,245),thick+2,cv2.LINE_AA)
      cv2.line(view,pt(marker),pt(tip),(139,130,23),thick,cv2.LINE_AA)
     cv2.line(view,pt(tips[0]),pt(tips[1]),(144,73,205),4,cv2.LINE_AA)
     for tip in tips:cv2.circle(view,pt(tip),8,(144,73,205),-1,cv2.LINE_AA)
    # Marker is drawn last, above both jaws and the distance line.
    marker_pt=tuple(np.rint(marker).astype(int))
    cv2.circle(view,marker_pt,13,(46,124,215),-1,cv2.LINE_AA)


   # Values use recorded physical units, independent of output image scaling.
   ratio=float(row['distance_over_initial'])
   if not np.isfinite(ratio):
    initial=next((float(q['distance_mm'])/float(q['distance_over_initial']) for q in rows if np.isfinite(float(q['distance_over_initial'])) and float(q['distance_over_initial'])>0),float(rows[0]['distance_mm']))
    ratio=float(row['distance_mm'])/initial
   aperture=float(row['lstm_mm'])
   layer=view.copy();cv2.rectangle(layer,(24,22),(660,138),(55,65,75),-1);cv2.addWeighted(layer,.82,view,.18,0,view)
   cv2.putText(view,'Distance L/L0',(44,67),0,.9,(255,255,255),2,cv2.LINE_AA)
   cv2.putText(view,f'{ratio*100:.1f}%',(430,67),0,.95,(60,225,255),2,cv2.LINE_AA)
   cv2.putText(view,'Predicted aperture',(44,114),0,.9,(255,255,255),2,cv2.LINE_AA)
   value=f'{aperture:.1f} mm' if np.isfinite(aperture) else '-- mm'
   cv2.putText(view,value,(430,114),0,.95,(170,120,245),2,cv2.LINE_AA)
   canvas=np.vstack([view,chart(float(m['video_time_s']))]);writer.write(canvas)
   if count==len(mapping)//2:cv2.imwrite(str(out/'preview.png'),canvas)
   count+=1
 finally:cap.release();writer.release()
 (out/'export_settings.json').write_text(json.dumps(dict(source=str(folder),frames=count,fps=fps,size=[width,vh+ch],raw_crop=[left,a.crop_y,640-left,a.crop_height],predictions='Saved CSV, unchanged; no new inference',overlay='LSTM aperture; schematic fingertip geometry'),indent=2),encoding='utf-8')
 print(out);print('frames',count,'size',width,vh+ch)
if __name__=='__main__':main()

