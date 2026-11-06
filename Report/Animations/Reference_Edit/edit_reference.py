from pathlib import Path
import cv2,numpy as np
from scipy.signal import savgol_filter
from PIL import Image,ImageDraw,ImageFont
source=Path('C:/Users/Di/Desktop/视频1.mp4')
out=Path('Report/Animations/Reference_Edit')
cap=cv2.VideoCapture(str(source))
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
tips=savgol_filter(np.asarray(tips),7,2,axis=0)
# Square center is fixed in this schematic.
object_center=np.array([293.,467.])
distances=np.array([np.linalg.norm(j-object_center) for j in joints])
ratios=distances/distances[0]
font=ImageFont.truetype('C:/Windows/Fonts/meiryo.ttc',21)
small=ImageFont.truetype('C:/Windows/Fonts/meiryo.ttc',17)
writer=cv2.VideoWriter(str(out/'reference_normalized_distance_trajectories.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1280,640))
assert writer.isOpened()
paths=[[],[]]
for index,(frame,joint,pair,ratio) in enumerate(zip(frames,joints,tips,ratios)):
 original=frame.copy()
 overlay=frame.copy()
 for side in [0,1]:
  paths[side].append(pair[side])
  if len(paths[side])>1:
   points=np.rint(paths[side]).astype(np.int32).reshape(-1,1,2)
   cv2.polylines(overlay,[points],False,(203,199,168),2,cv2.LINE_AA)
 frame=cv2.addWeighted(overlay,.8,frame,.2,0)
 # Keep the original fingers, marker and aperture line in front of traces.
 hsv=cv2.cvtColor(original,cv2.COLOR_BGR2HSV)
 color=(hsv[:,:,1]>65)&(hsv[:,:,2]>55)
 region=np.zeros(color.shape,bool);region[82:485,200:395]=True
 frame[color&region]=original[color&region]
 image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
 draw=ImageDraw.Draw(image)
 draw.text((65,520),f'lₜ / l₀ = {ratio:.2f}',font=font,fill='#DA692B')
 draw.line((68,573,102,573),fill='#A8C7CB',width=3)
 draw.text((113,559),'指先の移動軌跡',font=small,fill='#496773')
 frame=cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)
 writer.write(frame)
 if index==n//2:cv2.imwrite(str(out/'edited_preview.png'),frame)
writer.release()
cap=cv2.VideoCapture(str(out/'reference_normalized_distance_trajectories.mp4'))
assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==n
assert abs(cap.get(cv2.CAP_PROP_FPS)-fps)<1e-6
cap.set(cv2.CAP_PROP_POS_FRAMES,n-1);ok,last=cap.read();assert ok
cv2.imwrite(str(out/'edited_end.png'),last)
cap.release()
print('frames',n,'fps',fps,'ratio range',ratios.min(),ratios.max(),'end',ratios[-1])
print((out/'reference_normalized_distance_trajectories.mp4').resolve())

