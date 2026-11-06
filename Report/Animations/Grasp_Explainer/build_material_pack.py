from pathlib import Path
import math, shutil,zipfile
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
out=Path('Report/Animations/Grasp_Explainer/Jianying_Material_Pack')
for folder in ['01_background','02_transparent_layers','03_animation_green_screen','04_editable_text','05_source']:
 (out/folder).mkdir(parents=True,exist_ok=True)
W,H=1280,720
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
def smooth(x):
 x=max(0,min(1,x));return x*x*(3-2*x)
def aperture(q):
 return 10+90*smooth(q/.5) if q<.5 else 100-60*smooth((q-.5)/.5)
def layer(name,paint):
 im=Image.new('RGBA',(W,H),(0,0,0,0));paint(ImageDraw.Draw(im))
 im.save(out/'02_transparent_layers'/f'{name}.png')
Image.new('RGB',(W,H),'#F8FAFA').save(out/'01_background'/'background.png')
layer('object_30mm',lambda d:d.rounded_rectangle((524,404,566,446),radius=3,fill='#D5E4E7',outline='#142D3E',width=3))
def chart(d):
 for value in [0,25,50,75,100]:
  yy=540-value/110*290
  d.line((730,yy,1190,yy),fill='#E2E8E9',width=1)
 d.line((730,250,730,540,1190,540),fill='#142D3E',width=2)
layer('chart_grid_and_axes',chart)
layer('divider',lambda d:d.line((60,151,1220,151),fill='#E2E8E9',width=2))
layer('caption_strip',lambda d:d.rectangle((0,638,1280,720),fill='#EAF1F1'))
layer('joint_marker',lambda d:d.ellipse((86,416,104,434),fill='#EF9A40'))
layer('index_finger_initial',lambda d:d.line([(95,425),(161,418)],fill='#168B91',width=9))
layer('thumb_initial',lambda d:d.line([(95,425),(161,432)],fill='#168B91',width=9))
def trajectory(d,sign):
 pts=[]
 for q in np.linspace(0,1,180):
  l=300-255*smooth(q);pts.append((95+(300-l)*1.5+66,425+sign*aperture(q)*.7))
 d.line(pts,fill='#C6E2E1',width=3)
layer('index_trajectory',lambda d:trajectory(d,-1))
layer('thumb_trajectory',lambda d:trajectory(d,1))
# Separate animated layers on pure green. Use chroma key in Jianying.
writers={}
for name in ['index_finger','thumb','joint_marker','chart_curve']:
 writer=cv2.VideoWriter(str(out/'03_animation_green_screen'/f'{name}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),24,(W,H))
 assert writer.isOpened()
 writers[name]=writer
for n in range(480):
 sec=n/24;q=max(0,min(1,(sec-3)/13))
 l=300-255*smooth(q);x=95+(300-l)*1.5;half=aperture(q)*.7
 for name,writer in writers.items():
  im=Image.new('RGB',(W,H),(0,255,0));d=ImageDraw.Draw(im)
  if name in ['index_finger','thumb']:
   sign=-1 if name=='index_finger' else 1
   tip=(x+66,425+sign*half)
   d.line([(x,425),tip],fill='#168B91',width=9)
   d.ellipse((tip[0]-7,tip[1]-7,tip[0]+7,tip[1]+7),fill='#168B91')
  elif name=='joint_marker':
   d.ellipse((x-9,416,x+9,434),fill='#EF9A40')
  else:
   pts=[(730+z*460,540-aperture(z)/110*290) for z in np.linspace(0,q,max(2,int(q*180)))]
   d.line(pts,fill='#168B91',width=4)
   xx,yy=pts[-1];d.ellipse((xx-7,yy-7,xx+7,yy+7),fill='#168B91')
  writer.write(cv2.cvtColor(np.array(im),cv2.COLOR_RGB2BGR))
for w in writers.values():w.release()
srt='''1
00:00:00,000 --> 00:00:03,000
抓取时，手指会先张开，再闭合。

2
00:00:03,000 --> 00:00:09,000
距离逐渐缩短，手指开合同时变化。

3
00:00:09,000 --> 00:00:16,000
运动历史帮助描述动作随时间的变化。

4
00:00:16,000 --> 00:00:20,000
把动作和曲线放在一起，让预测看得见。
'''
(out/'04_editable_text'/'captions.srt').write_text(srt,encoding='utf-8-sig')
(out/'04_editable_text'/'copy.txt').write_text('''主标题
一只手，如何学会抓取？

副标题
从距离与物体尺寸，到手指开合

图形标签
俯视动作
手指开合 d
时间 / s
mm
30 mm

固定页脚
概念示意 · 轨迹与曲线为模拟数据
''',encoding='utf-8-sig')
# Editable vector reference with independent groups.
svg='''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">
<rect width="1280" height="720" fill="#F8FAFA"/>
<g id="title"><text x="60" y="85" font-family="Microsoft YaHei" font-size="42" fill="#142D3E">一只手，如何学会抓取？</text></g>
<g id="object"><rect x="524" y="404" width="42" height="42" rx="3" fill="#D5E4E7" stroke="#142D3E" stroke-width="3"/></g>
<g id="index"><path d="M95 425L161 418" fill="none" stroke="#168B91" stroke-width="9"/></g>
<g id="thumb"><path d="M95 425L161 432" fill="none" stroke="#168B91" stroke-width="9"/></g>
<g id="joint"><circle cx="95" cy="425" r="9" fill="#EF9A40"/></g>
<g id="chart"><path d="M730 250V540H1190" fill="none" stroke="#142D3E" stroke-width="2"/></g>
</svg>'''
(out/'05_source'/'editable_layout.svg').write_text(svg,encoding='utf-8')
shutil.copy2('Report/Animations/Grasp_Explainer/build_demo.py',out/'05_source'/'build_demo.py')
shutil.copy2('Report/Animations/Grasp_Explainer/grasp_explainer_demo.mp4',out/'preview.mp4')
(out/'剪映导入说明.txt').write_text('''剪映专业版分层素材包
画布 1280×720，24 FPS，时长20秒。全部动画是模拟示意。

1. 新建项目，导入 01_background、02_transparent_layers 和 03_animation_green_screen 中需要的素材。
2. 背景放最底层，长度设20秒。图片默认等比例完整显示在1280×720画布中，无需裁切。
3. 透明图层放背景上方，各自都能移动、缩放、旋转和调整时长。
4. 四个动画视频都从时间轴0秒开始，时长20秒。在“抠像/色度抠图”中选纯绿色，分别消除绿幕。先用较低强度，检查手指边缘。
5. 可选“轨迹”透明图层放在手指动画下方。静态的“初始手指、初始标记点”仅供自由重做关键帧，使用动画视频时不必叠加。
6. 从“文本/本地字幕”导入 04_editable_text/captions.srt，每句可直接在剪映编辑。主标题、副标题和标签内容见 copy.txt，用剪映文字添加即可。
7. 05_source/editable_layout.svg 是可编辑矢量排版参考。PNG图层在剪映中可作为独立素材编辑，SVG的内部路径需在支持SVG的矢量软件中编辑。
8. preview.mp4 是原演示片，用于对齐参考，可单独导入。

当前包是可导入剪映的素材包，并非剪映原生草稿。动画视频可做速度、裁切、位置和色彩编辑，内部每一帧的线条形状要通过源码重生成。
本机现有 draft_content.json 无法作为明文JSON读取，因此未生成未经验证的草稿。
''',encoding='utf-8-sig')
# Check all four animated files are complete.
for p in (out/'03_animation_green_screen').glob('*.mp4'):
 cap=cv2.VideoCapture(str(p));assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==480;cap.release()
archive=out.parent/'Jianying_Material_Pack.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in out.rglob('*'):
  if p.is_file():z.write(p,Path(out.name)/p.relative_to(out))
print('files',len([p for p in out.rglob('*') if p.is_file()]),'zip MB',round(archive.stat().st_size/1024**2,2))
print(archive.resolve())
