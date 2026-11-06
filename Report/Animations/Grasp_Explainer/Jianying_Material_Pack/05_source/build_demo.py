from pathlib import Path
import math
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

out=Path('Report/Animations/Grasp_Explainer')
W,H=1280,720
font_path='C:/Windows/Fonts/msyh.ttc'
fonts={s:ImageFont.truetype(font_path,s) for s in [16,18,20,24,28,34,42,54]}
navy='#142D3E'; muted='#73818A'; teal='#168B91'; orange='#EF9A40'; grid='#E2E8E9'
def txt(draw,xy,text,size=24,color=navy):
 draw.text(xy,text,font=fonts[size],fill=color)
def smooth(x):
 x=min(1,max(0,x))
 return x*x*(3-2*x)
def aperture(q):
 if q<.5:return 10+90*smooth(q/.5)
 return 100-60*smooth((q-.5)/.5)
def state(q):
 if q<.12:return '准备接近'
 if q<.5:return '接近，同时张开'
 if q<.88:return '继续接近，逐渐闭合'
 return '抓住，保持'
def frame(seconds):
 im=Image.new('RGB',(W,H),'#F8FAFA');d=ImageDraw.Draw(im)
 txt(d,(60,36),'一只手，如何学会抓取？',42)
 txt(d,(62,100),'从距离与物体尺寸，到手指开合',24,muted)
 d.line((60,151,1220,151),fill=grid,width=2)
 q=min(1,max(0,(seconds-3)/13))
 l=300-255*smooth(q)
 opening=aperture(q)
 jointx=95+(300-l)*1.5
 cy=425
 tipx=jointx+66
 half=opening*1.4/2
 cube_x=545
 # Object
 d.rounded_rectangle((cube_x-21,cy-21,cube_x+21,cy+21),radius=3,fill='#D5E4E7',outline=navy,width=3)
 txt(d,(cube_x-43,cy+45),'30 mm',20)
 # trajectory hints
 for sign in [-1,1]:
  points=[]
  for z in np.linspace(0,q,max(2,int(q*120))):
   ll=300-255*smooth(z)
   points.append((95+(300-ll)*1.5+66,cy+sign*aperture(z)*1.4/2))
  if len(points)>1:d.line(points,fill='#C6E2E1',width=3)
 # Finger diagram
 top=(tipx,cy-half);bottom=(tipx,cy+half)
 d.line([(jointx,cy),top],fill=teal,width=9)
 d.line([(jointx,cy),bottom],fill=teal,width=9)
 for point in [top,bottom]:
  x,y=point;d.ellipse((x-7,y-7,x+7,y+7),fill=teal)
 d.ellipse((jointx-9,cy-9,jointx+9,cy+9),fill=orange)
 # distance segment and aperture
 d.line((jointx,cy+110,cube_x,cy+110),fill=orange,width=2)
 for x in [jointx,cube_x]:d.line((x,cy+103,x,cy+117),fill=orange,width=2)
 txt(d,((jointx+cube_x)/2-45,cy+120),f'l = {l:.0f} mm',18,orange)
 d.line((tipx+17,cy-half,tipx+17,cy+half),fill=teal,width=2)
 txt(d,(68,205),'俯视动作',24)
 txt(d,(68,247),state(q),28,teal)
 # Chart
 left,topc,right,bottomc=730,250,1190,540
 for value in [0,25,50,75,100]:
  yy=bottomc-value/110*(bottomc-topc)
  d.line((left,yy,right,yy),fill=grid,width=1)
  txt(d,(left-42,yy-11),str(value),16,muted)
 d.line((left,topc,left,bottomc,right,bottomc),fill=navy,width=2)
 for second in range(4):
  xx=left+second/3*(right-left)
  txt(d,(xx-5,bottomc+12),str(second),16,muted)
 txt(d,(730,205),'手指开合 d',24)
 txt(d,(704,222),'mm',16,muted)
 txt(d,(940,581),'时间 / s',18,muted)
 points=[(left+z*(right-left),bottomc-aperture(z)/110*(bottomc-topc)) for z in np.linspace(0,q,max(2,int(q*180)))]
 d.line(points,fill=teal,width=4)
 xx=left+q*(right-left);yy=bottomc-opening/110*(bottomc-topc)
 d.ellipse((xx-7,yy-7,xx+7,yy+7),fill=teal)
 txt(d,(945,158),f'd = {opening:.0f} mm',28,teal)
 # Progressive explanation
 if seconds<3:
  message='抓取时，手指会先张开，再闭合。'
 elif seconds<9:
  message='距离逐渐缩短，手指开合同时变化。'
 elif seconds<16:
  message='运动历史帮助描述动作随时间的变化。'
 else:
  message='把动作和曲线放在一起，让预测看得见。'
 d.rectangle((0,638,W,720),fill='#EAF1F1')
 txt(d,(60,651),message,24)
 txt(d,(62,690),'概念示意 · 轨迹与曲线为模拟数据',16,muted)
 if seconds<1:
  overlay=Image.new('RGB',(W,H),'#F8FAFA')
  im=Image.blend(overlay,im,smooth(seconds))
 return im

writer=cv2.VideoWriter(str(out/'grasp_explainer_demo.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),24,(W,H))
assert writer.isOpened()
for n in range(20*24):
 im=frame(n/24)
 writer.write(cv2.cvtColor(np.array(im),cv2.COLOR_RGB2BGR))
 if n==240:im.save(out/'preview.png')
writer.release()
cap=cv2.VideoCapture(str(out/'grasp_explainer_demo.mp4'))
print('frames',int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),'fps',cap.get(cv2.CAP_PROP_FPS))
assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==480
cap.release()
print((out/'grasp_explainer_demo.mp4').resolve())
