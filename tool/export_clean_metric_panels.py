"""Export separate clean metric panels from saved Test metrics, no retraining."""
import os,csv
from pathlib import Path
os.environ['MPLCONFIGDIR']=str(Path(__file__).parent/'.mplconfig')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont('C:/Windows/Fonts/meiryo.ttc')
plt.rcParams.update({'font.family':'Meiryo','font.size':12,'xtick.labelsize':12,'ytick.labelsize':11,'axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'none'})
P=Path(__file__).resolve().parents[1]/'Data/1029_D1/report_metric_figures'
MODELS=['Function','MLP 5f','LSTM Full'];COLORS=['#3485b8','#e59a32','#519e58'];SIZES=['20 mm','30 mm','40 mm']
def main():
 with (P/'test_metrics_three_objects.csv').open(encoding='utf-8-sig',newline='') as f:rows={(r['object'],r['model']):r for r in csv.DictReader(f)}
 dest=P/'clean';dest.mkdir(exist_ok=True)
 for key,name in [('mae_mm','MAE'),('rmse_mm','RMSE'),('r2','R2')]:
  fig,ax=plt.subplots(figsize=(5.2,4.1),facecolor='white');fig.subplots_adjust(left=.12,right=.985,bottom=.14,top=.97)
  labels=[];maximum=0;pos=np.arange(3);width=.23
  for j,(model,color) in enumerate(zip(MODELS,COLORS)):
   values=[float(rows[size,model][key]) for size in SIZES];maximum=max(maximum,max(values))
   bars=ax.bar(pos+(j-1)*width,values,width=width*.94,color=color,zorder=3)
   labels.extend(ax.bar_label(bars,fmt='%.3f' if key=='r2' else '%.2f',padding=4,fontsize=9))
  ax.set_xticks(pos,SIZES);ax.set_xlim(-.55,2.55);ax.set_ylim(0,1.10 if key=='r2' else np.ceil(maximum*1.22))
  if key=='r2':ax.set_yticks(np.arange(0,1.01,.2))
  ax.grid(axis='y',alpha=.2,zorder=0);ax.spines[['top','right']].set_visible(False);ax.tick_params(axis='x',length=0,pad=8)
  fig.canvas.draw();placed=[]
  for text in sorted(labels,key=lambda t:t.xy[0]):
   for _ in range(30):
    box=text.get_window_extent(fig.canvas.get_renderer()).expanded(1.05,1.15)
    if not any(box.overlaps(other) for other in placed):break
    x,y=text.get_position();text.set_position((x,y+5));fig.canvas.draw()
   placed.append(text.get_window_extent(fig.canvas.get_renderer()).expanded(1.05,1.15))
  assert not ax.get_title() and ax.get_legend() is None and len(fig.axes)==1
  for ext in ['png','pdf','svg']:fig.savefig(dest/f'{name}.{ext}',dpi=300,facecolor='white')
  plt.close(fig)
 print('Saved separate clean MAE, RMSE and R2 panels:',dest)
if __name__=='__main__':main()
