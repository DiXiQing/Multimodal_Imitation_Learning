"""Generate the English reference animation, then open the video player.

Requires the original 视频1.mp4. Run --help for input/output options.
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import argparse
import os
from pathlib import Path
import cv2,numpy as np
from scipy.interpolate import BSpline, make_lsq_spline
from PIL import Image,ImageDraw,ImageFont
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, default=Path.home() / 'Desktop' / 'grasp_explainer_demo.mp4')
parser.add_argument('--output-dir', type=Path,
                    default=Path(__file__).resolve().parents[1] / 'Report/Animations/Reference_Edit')
parser.add_argument('--no-open', action='store_true', help='Generate without opening the player')
args = parser.parse_args()
source = args.source.resolve()
if not source.is_file():
    parser.error(f'Original video missing: {source}')
out = args.output_dir.resolve()
out.mkdir(parents=True, exist_ok=True)
print(f'Reading original video: {source}', flush=True)
cap=cv2.VideoCapture(str(source))
if not cap.isOpened():
    raise RuntimeError(f'Cannot read video: {source}')
if int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) != 1280 or int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) != 640:
    raise ValueError('This script uses coordinates from the 1280 x 640 reference video.')
fps=cap.get(cv2.CAP_PROP_FPS)
n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frames=[]; joints=[];tips=[]
while True:
 ok,frame=cap.read()
 if not ok:break
 hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
 orange=cv2.inRange(hsv,np.array([4,130,120]),np.array([24,255,255]))
 orange[:82]=0;orange[:, :240]=0;orange[:,350:]=0;orange[450:]=0
 count,labels,stats,centers=cv2.connectedComponentsWithStats(orange)
 options=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>100 and stats[i,cv2.CC_STAT_WIDTH]>10]
 if not options:raise ValueError(f'joint missing {len(frames)}')
 i=max(options,key=lambda i:stats[i,cv2.CC_STAT_AREA])
 joint=centers[i]
 teal=cv2.inRange(hsv,np.array([75,150,70]),np.array([104,255,255]))
 teal[:100]=0;teal[:, :200]=0;teal[:,395:]=0;teal[470:]=0
 ys,xs=np.where(teal>0)
 if not len(ys):raise ValueError('tips missing')
 pair=[]
 for side in [-1,1]:
  take=xs<joint[0] if side==-1 else xs>joint[0]
  side_y=ys[take];side_x=xs[take]
  low=side_y.max()
  cap_pixels=side_x[side_y>=low-2]
  pair.append((float(np.mean(cap_pixels)),float(low-1)))
 tips.append(tuple(pair))
 joints.append(joint);frames.append(frame)
cap.release()
assert len(frames)==n
tips=np.asarray(tips)
parameter=np.linspace(0,1,n)
knots=np.r_[np.zeros(4),np.linspace(0,1,6)[1:-1],np.ones(4)]
trajectory_splines=[]
for side in range(2):
 trajectory_splines.append(make_lsq_spline(parameter,tips[:,side,:],knots,k=3))
tips=np.stack([spline(parameter) for spline in trajectory_splines],axis=1)
# Square center is fixed in this schematic.
object_center=np.array([293.,467.])
distances=np.array([np.linalg.norm(j-object_center) for j in joints])
ratios=distances/distances[0]
font=ImageFont.truetype('C:/Windows/Fonts/meiryo.ttc',21)
small=ImageFont.truetype('C:/Windows/Fonts/meiryo.ttc',17)
writer=cv2.VideoWriter(str(out/'reference_english_smooth_trajectories.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1280,640))
assert writer.isOpened()
paths=[[],[]]
for index,(frame,joint,pair,ratio) in enumerate(zip(frames,joints,tips,ratios)):
 original=frame.copy()
 overlay=frame.copy()
 for side in [0,1]:
  paths[side]=trajectory_splines[side](np.linspace(0,index/(n-1),max(2,index*6+1))).tolist()
  if len(paths[side])>1:
   points=np.rint(np.asarray(paths[side])*16).astype(np.int32).reshape(-1,1,2)
   cv2.polylines(overlay,[points],False,(203,199,168),2,cv2.LINE_AA,shift=4)
 frame=cv2.addWeighted(overlay,.8,frame,.2,0)
 # Keep the original fingers, marker and aperture line in front of traces.
 hsv=cv2.cvtColor(original,cv2.COLOR_BGR2HSV)
 color=(hsv[:,:,1]>65)&(hsv[:,:,2]>55)
 region=np.zeros(color.shape,bool);region[82:485,200:395]=True
 frame[color&region]=original[color&region]
 image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
 draw=ImageDraw.Draw(image)
 english=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',22)
 heading=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',25)
 note=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
 # Clear title and legend text while retaining colored legend symbols.
 for box in [(125,16,474,57),(200,57,475,89),(200,88,475,117),(200,117,475,149),
             (804,15,1225,55),(803,569,1245,606),(782,610,1279,639),
             (710,58,995,91)]:
  draw.rectangle(box,fill='white')
 # Clear annotation lettering, then restore the underlying teal curve.
 arr=np.array(image)
 for x1,y1,x2,y2 in [(620,377,900,412),(1070,432,1265,470)]:
  arr[y1:y2,x1:x2]=255
  curve_mask=cv2.inRange(hsv[y1:y2,x1:x2],np.array([75,150,70]),np.array([104,255,255]))>0
  original_rgb=cv2.cvtColor(original[y1:y2,x1:x2],cv2.COLOR_BGR2RGB)
  arr[y1:y2,x1:x2][curve_mask]=original_rgb[curve_mask]
 image=Image.fromarray(arr)
 draw=ImageDraw.Draw(image)
 draw.text((130,22),'Top-view grasp schematic',font=heading,fill='#263E5D')
 draw.text((204,58),'Index-finger base',font=english,fill='#DA692B')
 draw.text((204,89),'Finger aperture d',font=english,fill='#B84059')
 draw.text((204,119),'Distance L',font=english,fill='#DA692B')
 draw.text((813,22),'Aperture-distance trajectory',font=heading,fill='#263E5D')
 draw.text((811,572),'Normalized distance l = L / L0',font=english,fill='#111111')
 draw.text((797,615),'Schematic animation (simulated data)',font=note,fill='#888888')
 draw.text((625,381),'1  Start closed',font=english,fill='#263E5D')
 if index>=n//4:
  draw.text((714,62),'2  Maximum aperture',font=english,fill='#263E5D')
 if index>=int(n*.75):
  draw.text((1073,438),'3  Close to grasp',font=english,fill='#263E5D')
 # Replace the vertical Japanese axis label with rotated English text.
 draw.rectangle((505,180,543,405),fill='white')
 axis=Image.new('RGBA',(250,32),(255,255,255,0))
 ImageDraw.Draw(axis).text((2,2),'Finger aperture d (mm)',font=english,fill='#111111')
 axis=axis.rotate(90,expand=True)
 image.paste(axis,(507,188),axis)
 # Repaint the cube face and its center marker.
 draw.rectangle((278,451,307,481),fill='#333333')
 draw.ellipse((289,463,297,471),fill='white')
 draw.text((65,520),f'l(t) = {ratio:.2f}',font=font,fill='#DA692B')
 draw.line((68,573,102,573),fill='#A8C7CB',width=3)
 draw.text((113,559),'Fingertip trajectories',font=note,fill='#496773')
 frame=cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
 if joint[1]<165:
  preserve=np.zeros(frame.shape[:2],bool)
  preserve[max(82,int(joint[1])-13):200,275:313]=True
  hand_color=(hsv[:,:,1]>130)&(hsv[:,:,2]>70)
  frame[preserve&hand_color]=original[preserve&hand_color]
 writer.write(frame)
 if index==n//2:cv2.imwrite(str(out/'edited_preview.png'),frame)
writer.release()
cap=cv2.VideoCapture(str(out/'reference_english_smooth_trajectories.mp4'))
assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==n
assert abs(cap.get(cv2.CAP_PROP_FPS)-fps)<1e-6
cap.set(cv2.CAP_PROP_POS_FRAMES,n-1);ok,last=cap.read();assert ok
cv2.imwrite(str(out/'edited_end.png'),last)
cap.release()
print('frames',n,'fps',fps,'ratio range',ratios.min(),ratios.max(),'end',ratios[-1])
print((out/'reference_english_smooth_trajectories.mp4').resolve())


if not args.no_open:
    os.startfile(str(out / 'reference_english_smooth_trajectories.mp4'))
