from pathlib import Path
import csv
import numpy as np
from PIL import Image,ImageOps,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
ROOT=Path(r'C:\MineApp\Code\Multimodal_Imitation_Learning');OUT=ROOT/'Report/Realtime_Slide_Mockups';OUT.mkdir(exist_ok=True)
NAVY='#102e45';TEAL='#188b96';BLUE='#56B4E9';ORANGE='#FFB366';GRAY='#60717e'
plt.rcParams.update({'font.family':'DejaVu Sans'})
def slide(title,sub,num):
 fig=plt.figure(figsize=(10,10),dpi=160,facecolor='white')
 fig.text(.065,.925,title,fontsize=28,fontfamily='DejaVu Serif',fontweight='bold',color=NAVY)
 fig.text(.068,.88,sub,fontsize=12,color=GRAY)
 fig.text(.065,.035,'Saga University | Fukuda Laboratory',fontsize=9,color=TEAL)
 fig.text(.94,.035,str(num),fontsize=10,color=GRAY,ha='right')
 return fig

def text(fig,x,y,s,size=17,color=NAVY,weight='normal'):
 fig.text(x,y,s,fontsize=size,color=color,weight=weight,va='top',linespacing=1.6)
def save(fig,name):
 fig.savefig(OUT/name);plt.close(fig)
fig=slide('Real-Time Experiment','From offline learning to online finger-aperture estimation',1)
text(fig,.07,.79,'3 cube sizes',22,TEAL,'bold');text(fig,.07,.735,'20 / 30 / 40 mm',19)
text(fig,.53,.79,'3 initial distances',22,TEAL,'bold');text(fig,.53,.735,'Approximately 250 / 300 / 350 mm',16)
text(fig,.07,.64,'3 movement durations',22,TEAL,'bold');text(fig,.07,.585,'Approximately 1 / 2 / 3 s',18)
text(fig,.53,.64,'81 trials per participant',23,TEAL,'bold');text(fig,.53,.58,'27 conditions × 3 repetitions\nProtocol: randomized order',15)
ax=fig.add_axes([.07,.32,.86,.15]);ax.axis('off')
for i,(label,fill) in enumerate([('Camera','#edf7fb'),('Feature extraction','#edf7fb'),('Three models','#e9f5f2'),('Aperture estimate','#e9f5f2')]):
 x=i*.255;ax.add_patch(Rectangle((x,.2),.225,.65,facecolor=fill,edgecolor=TEAL,lw=1.2));ax.text(x+.1125,.525,label,ha='center',va='center',fontsize=11)
 if i<3:ax.annotate('',xy=(x+.25,.525),xytext=(x+.228,.525),arrowprops=dict(arrowstyle='->',color=TEAL))
text(fig,.07,.245,'Function / MLP (5 frames) / LSTM (full history)',18,NAVY,'bold')
text(fig,.07,.185,'All three models run simultaneously on the same trial.',16)
text(fig,.07,.10,'Analysis set: 81 selected trials per participant.\nModel source and velocity processing must be stated for each recording.',10,GRAY)
save(fig,'01_experiment.png')
fig=slide('Live Prediction Demonstration','Synchronized overhead view and model outputs',2)
screen=Image.open(r'C:\Users\Di\AppData\Local\Temp\codex-clipboard-b752f5c8-6579-4e1b-932b-08f34d460d1d.png').convert('RGB')
# Extract the supplied real camera-and-curve panel; no synthetic experimental footage.
panel=screen.crop((66,116,895,349))
ax=fig.add_axes([.065,.39,.87,.40]);ax.imshow(panel);ax.axis('off')
text(fig,.07,.79,'OVERHEAD VIDEO + LIVE ESTIMATES',14,TEAL,'bold')
for x,label in [(.10,'01  Rest'),(.39,'02  Approach & grasp'),(.77,'03  Rest')]:text(fig,x,.34,label,15,TEAL,'bold')
text(fig,.07,.245,'One trial. Three models. Synchronized outputs.',21,NAVY,'bold')
text(fig,.07,.17,'Show the full sequence to inspect onset, tracking and rest behavior.',14)
text(fig,.07,.10,'Layout preview: replace this still with the synchronized video.\nLabel the participant, cube size and trained-model source.',10,GRAY)
save(fig,'02_live_demonstration.png')
with (ROOT/'Report/Two_Participant_Comparison/comparison_values.csv').open(encoding='utf-8-sig') as f:records=list(csv.DictReader(f))
fig=slide('Real-Time Prediction Accuracy','n = 2 participants | 81 selected trials each | Mean ± SD',3)
models=['Function','MLP 5f','LSTM Full'];x=np.arange(3)
for j,(metric,title,limit) in enumerate([('mae','MAE (mm)',12),('rmse','RMSE (mm)',15),('r2','R²',1.18)]):
 ax=fig.add_axes([.075+j*.305,.37,.255,.37])
 for off,person,color,label in [(-.18,'Di',BLUE,'Participant 1'),(.18,'Song',ORANGE,'Participant 2')]:
  g=[next(r for r in records if r['evaluation']=='realtime' and r['participant']==person and r['model']==m and r['metric']==metric) for m in models]
  means=np.array([float(r['mean']) for r in g]);sd=np.array([float(r['sd']) for r in g])
  ax.bar(x+off,means,.35,yerr=sd,color=color,capsize=3,label=label,error_kw={'elinewidth':1})
  for pos,v,s in zip(x+off,means,sd):ax.text(pos,v+s+limit*.018,f'{v:.3f}' if metric=='r2' else f'{v:.2f}',ha='center',fontsize=7)
 ax.set_title(title,fontsize=15,pad=13);ax.set_xticks(x,['Function','MLP 5f','LSTM'],fontsize=9);ax.tick_params(labelsize=9);ax.set_ylim(0,limit);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
 if j==0:handles,labels=ax.get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.83),ncol=2,frameon=False,fontsize=13)
text(fig,.08,.26,'LSTM had the lowest mean MAE and RMSE\nin both selected participant datasets.',21,NAVY,'bold')
text(fig,.08,.13,'Existing real-time recordings; not results from the new LinearFit5 training.\nFull results by cube size can be placed in the appendix.',10,GRAY)
save(fig,'03_accuracy.png')
fig=slide('Processing Time and Stability','Separate computation time from camera-to-output latency',4)
text(fig,.07,.79,'18.78 ms',42,TEAL,'bold');text(fig,.07,.71,'Mean measured processing time per frame',14)
text(fig,.57,.79,'Participant 2',18,NAVY,'bold');text(fig,.57,.745,'5,505 recorded frames\nMedian: 17.49 ms',14)
ax=fig.add_axes([.27,.40,.64,.22]);labels=['Hand / object detection','Function','MLP','LSTM'];values=[16.739,.034,.582,.413]
ax.barh(np.arange(4),values,color=[BLUE,ORANGE,ORANGE,ORANGE]);ax.set_yticks(np.arange(4),labels,fontsize=11);ax.invert_yaxis();ax.set_xlabel('Recorded component time (ms)',fontsize=11);ax.set_xlim(0,19)
for i,v in enumerate(values):ax.text(v+.2,i,f'{v:.3f}',va='center',fontsize=10)
ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
text(fig,.08,.285,'Rest-phase stability',21,TEAL,'bold')
text(fig,.08,.225,'Add a measured rest-only trace and prediction SD\nfrom the demonstration recording.',16)
text(fig,.08,.12,'Stability analysis pending. Component times do not sum to total processing time.\nCamera-to-output latency was not established by these processing timers.',10,GRAY)
save(fig,'04_timing_stability.png')
imgs=[Image.open(OUT/f'{i:02d}_{n}.png').convert('RGB') for i,n in [(1,'experiment'),(2,'live_demonstration'),(3,'accuracy'),(4,'timing_stability')]]
thumbs=[im.resize((650,650)) for im in imgs];board=Image.new('RGB',(1330,1330),'#e7eaed')
for im,(x,y) in zip(thumbs,[(10,10),(670,10),(10,670),(670,670)]):board.paste(im,(x,y))
board.save(OUT/'overview.png')
print(OUT)

