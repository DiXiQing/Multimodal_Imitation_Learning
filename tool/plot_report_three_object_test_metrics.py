"""Slide-ready three-metric comparison of three models across three objects."""
import os,csv,json,hashlib
from pathlib import Path
os.environ['MPLCONFIGDIR']=str(Path(__file__).parent/'.mplconfig')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
font_manager.fontManager.addfont('C:/Windows/Fonts/meiryo.ttc')
font_manager.fontManager.addfont('C:/Windows/Fonts/meiryob.ttc')
plt.rcParams.update({'font.family':'Meiryo','font.size':12,'axes.labelsize':12,'axes.titlesize':17,'xtick.labelsize':12,'ytick.labelsize':11,'axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'none'})
P=Path(__file__).resolve().parents[1]/'Data/1029_D1'
OUT=P/'report_metric_figures';OUT.mkdir(exist_ok=True)
MODELS=['Function','MLP 5f','LSTM Full'];COLORS=['#3485b8','#e59a32','#519e58']
SOURCES=[('20mm_red','20 mm'),('30mm_blue','30 mm'),('40mm_black','40 mm')]
def main():
 data={};audit=[];metrics=[]
 for folder,label in SOURCES:
  runs=sorted(p for p in (P/folder/'Result').glob('l0_normalized_function_mlp5_lstmfull_400ep_*') if (p/'overall_metrics.csv').is_file())
  run=runs[-1];file=run/'overall_metrics.csv'
  with file.open(encoding='utf-8-sig',newline='') as f:rows={r['model']:r for r in csv.DictReader(f) if r['split']=='Test'}
  assert set(rows)==set(MODELS);assert len({r['frames'] for r in rows.values()})==1
  data[label]=rows;audit.append(dict(object=label,result_directory=str(run),overall_metrics_sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
  for model in MODELS:metrics.append(dict(object=label,model=model,frames=int(rows[model]['frames']),mae_mm=float(rows[model]['mae_mm']),rmse_mm=float(rows[model]['rmse_mm']),r2=float(rows[model]['r2'])))
 fig,axes=plt.subplots(1,3,figsize=(15.3,4.5),facecolor='white')
 fig.subplots_adjust(left=.045,right=.99,bottom=.14,top=.78,wspace=.26)
 pos=np.arange(3);width=.23
 for ax,key,title in zip(axes,['mae_mm','rmse_mm','r2'],['MAE [mm] ↓','RMSE [mm] ↓','R² ↑']):
  largest=0; annotations=[]
  for j,(model,color) in enumerate(zip(MODELS,COLORS)):
   values=[float(data[label][model][key]) for _,label in SOURCES];largest=max(largest,max(values))
   bars=ax.bar(pos+(j-1)*width,values,width=width*.94,color=color,zorder=3)
   annotations.extend(ax.bar_label(bars,fmt='%.3f' if key=='r2' else '%.2f',padding=4,fontsize=9))
  ax.set_xticks(pos,[label for _,label in SOURCES]);ax.set_title(title,pad=13,fontweight='bold')
  ax.set_ylim(0,1.10 if key=='r2' else np.ceil(largest*1.22));ax.set_xlim(-.55,2.55)
  ax.grid(axis='y',alpha=.2,zorder=0);ax.spines[['top','right']].set_visible(False);ax.tick_params(axis='x',length=0,pad=8)
  if key=='r2':ax.set_yticks(np.arange(0,1.01,.2))
  # Move labels only when their rendered text boxes collide.
  fig.canvas.draw();placed=[]
  for text in sorted(annotations,key=lambda t:t.xy[0]):
   for attempt in range(30):
    box=text.get_window_extent(fig.canvas.get_renderer()).expanded(1.05,1.15)
    if not any(box.overlaps(other) for other in placed):break
    x,y=text.get_position();text.set_position((x,y+5));fig.canvas.draw()
   placed.append(text.get_window_extent(fig.canvas.get_renderer()).expanded(1.05,1.15))
 fig.legend(handles=[Patch(facecolor=c,label=m) for m,c in zip(MODELS,COLORS)],loc='upper center',bbox_to_anchor=(.5,.99),ncol=3,frameon=False,fontsize=13,handlelength=1.5,columnspacing=2.5)
 for ext in ['png','pdf','svg']:fig.savefig(OUT/f'test_metrics_three_objects.{ext}',dpi=300,facecolor='white')
 plt.close(fig)
 with (OUT/'test_metrics_three_objects.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
 (OUT/'manifest.json').write_text(json.dumps(dict(sources=audit,scope='Test only; pooled common valid frames for the three models within each object',layout='Metric panels MAE/RMSE/R2; object size groups; model colours fixed across all panels',labels='MAE/RMSE 2 decimals in mm, R2 3 decimals unitless',error_bars='None: pooled test metrics, not mean +/- per-trial SD',training_performed=False),indent=2),encoding='utf-8')
 print(OUT/'test_metrics_three_objects.png')
if __name__=='__main__':main()
