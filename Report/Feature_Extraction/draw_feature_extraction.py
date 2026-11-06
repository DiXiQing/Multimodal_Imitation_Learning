from pathlib import Path
import ast,csv,json
import cv2,numpy as np,mediapipe as mp
from PIL import Image,ImageDraw,ImageFont
root=Path(__file__).resolve().parents[2]
folder=root/'Data/1101_D/20mm/20261101_195810'
out=root/'Report/Feature_Extraction'
original=cv2.imread(str(folder/'frames/frame_005517.jpg'))
crop=cv2.imread(str(folder/'frames/frame_005517(1).jpg'))
score=cv2.matchTemplate(original,crop,cv2.TM_CCOEFF_NORMED)
_,quality,_,offset=cv2.minMaxLoc(score)
print('crop',offset,'match',quality)
if quality<.85:raise ValueError('Could not align supplied crop')
with (folder/'camera.csv').open(encoding='utf-8-sig') as f:
 row=next(r for r in csv.DictReader(f) if r['frame']=='5517')
settings=json.loads((folder/'settings.json').read_text(encoding='utf-8-sig'))
with mp.solutions.hands.Hands(static_image_mode=True,max_num_hands=1,min_detection_confidence=.4) as hands:
 result=hands.process(cv2.cvtColor(original,cv2.COLOR_BGR2RGB))
 if not result.multi_hand_landmarks:raise ValueError('Hand not detected')
 marks=result.multi_hand_landmarks[0]
 coords=np.array([[m.x*original.shape[1],m.y*original.shape[0]] for m in marks.landmark])
for idx,prefix in [(0,'wrist'),(4,'thumb'),(5,'index_mcp'),(8,'index')]:
 coords[idx]=[float(row[prefix+'_x']),float(row[prefix+'_y'])]
# Reuse the recording script's exact colour and contour detector.
tree=ast.parse((root/'Program/Record.py').read_text(encoding='utf-8-sig'))
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['color_ranges','detect_cube']]
namespace={'cv2':cv2,'np':np}
exec(compile(ast.Module(body=functions,type_ignores=[]),'Record detector','exec'),namespace)
(_,box),_=namespace['detect_cube'](original,black_max=settings['v'],saturation_max=settings['s'],min_area=settings['area'],color='black',max_area=settings['max_area'])
if box is None:raise ValueError('Object not detected')
factor=5
image=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)).resize((crop.shape[1]*factor,crop.shape[0]*factor),Image.Resampling.LANCZOS)
d=ImageDraw.Draw(image)
def xy(point):
 return tuple((np.asarray(point)-offset)*factor)
def line(a,b,color,width=3):
 d.line([xy(a),xy(b)],fill=color,width=width)
def dot(p,r,color):
 x,y=xy(p);d.ellipse((x-r,y-r,x+r,y+r),fill=color)
for a,b in mp.solutions.hands.HAND_CONNECTIONS:line(coords[a],coords[b],'#E8F1EB',2)
for i,p in enumerate(coords):
 dot(p,4,'#72D69A')
x,y,w,h=box
d.rectangle([xy((x,y)),xy((x+w,y+h))],outline='#57E363',width=3)
obj=np.array([float(row['object_x']),float(row['object_y'])])
line(obj,coords[5],'#F8A239',4)
line(coords[4],coords[8],'#42C9F5',4)
dot(obj,6,'#FF665C')
dot(coords[5],7,'#F8A239')
dot(coords[4],6,'#42C9F5');dot(coords[8],6,'#42C9F5')
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19)
for idx in [0,4,5,8]:
 xx,yy=xy(coords[idx])
 d.text((xx+7,yy-23),str(idx),font=font,fill='white',stroke_width=2,stroke_fill='#26313A')
xx,yy=xy((obj+coords[5])/2)
d.text((xx-8,yy-30),'L',font=font,fill='#F8A239',stroke_width=1,stroke_fill='#26313A')
xx,yy=xy((coords[4]+coords[8])/2)
d.text((xx+8,yy),'d',font=font,fill='#42C9F5',stroke_width=1,stroke_fill='#26313A')
image.save(out/'feature_extraction_overlay.png')
canvas=Image.new('RGB',(1320,720),'white')
canvas.paste(image,(35,75))
draw=ImageDraw.Draw(canvas)
title=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',30)
body=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',23)
small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
draw.text((35,22),'Feature Extraction',font=title,fill='#183348')
labels=[
 ('#72D69A','Hand landmarks and connections'),
 ('#57E363','Detected object bounding box'),
 ('#FF665C','Object center'),
 ('#F8A239','Index MCP (landmark 5)'),
 ('#42C9F5','Fingertips (landmarks 4 and 8)'),
 ('#F8A239',f'L = {float(row["index_mcp_object_distance_mm"]):.1f} mm'),
 ('#42C9F5',f'd = {float(row["finger_width_mm"]):.1f} mm'),
]
for i,(color,text) in enumerate(labels):
 yy=110+i*58
 draw.ellipse((870,yy+5,882,yy+17),fill=color)
 draw.text((897,yy),text,font=body,fill='#183348')
draw.text((870,540),'L: index-finger base to object center',font=small,fill='#526572')
draw.text((870,572),'d: thumb tip to index fingertip',font=small,fill='#526572')
draw.text((35,680),'Frame 5517. Key measurement points use saved CSV coordinates; remaining landmarks are re-detected.',font=small,fill='#526572')
canvas.save(out/'feature_extraction_explained.png')
(out/'source.json').write_text(json.dumps(dict(frame=5517,crop_offset=offset,match=quality,box=box,key_points='saved CSV',other_landmarks='MediaPipe re-detection'),indent=2),encoding='utf-8')
print(out)
