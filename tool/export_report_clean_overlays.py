"""Export three independent report figures, duration-coloured, mm and l/l0 versions."""
import os,csv,json,re,hashlib
from pathlib import Path
os.environ['MPLCONFIGDIR']=str(Path(__file__).parent/'.mplconfig')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, colors
P=Path(__file__).resolve().parents[1]/'Data/1029_D1'
OUT=P/'report_overlay_figures'/'clean'
font_manager.fontManager.addfont('C:/Windows/Fonts/meiryo.ttc')
font_manager.fontManager.addfont('C:/Windows/Fonts/meiryob.ttc')
plt.rcParams.update({'font.family':'Meiryo','axes.unicode_minus':False,'font.size':13,'axes.labelsize':16,'axes.titlesize':20,'xtick.labelsize':12,'ytick.labelsize':12,'pdf.fonttype':42,'svg.fonttype':'none'})
SPECS=[('20mm_red',20),('30mm_blue',30),('40mm_black',40)]
def main():
 groups=[];audit=[]
 for folder,size in SPECS:
  root=P/folder
  if (root/'Train').is_dir() and (root/'Test').is_dir():
   paths=sorted(f for split in ['Train','Test'] for f in (root/split).iterdir() if f.is_dir() and re.fullmatch(r'grasp_\d+',f.name));source='Train + Test (one copy per trial)'
  else:paths=sorted(f for f in root.iterdir() if f.is_dir() and re.fullmatch(r'grasp_\d+',f.name));source='Active root trials'
  assert len(paths)==100,(folder,len(paths))
  assert len({f.name for f in paths})==100
  group=[]
  for trial in paths:
   with (trial/'camera.csv').open(encoding='utf-8-sig',newline='') as f:r=list(csv.DictReader(f))
   meta=json.loads((trial/'segment.json').read_text());settings=json.loads((trial/'settings.json').read_text(encoding='utf-8-sig'));scale=float(settings['pixel_calibration']['mm_per_px']);assert scale>0
   v=lambda k:np.array([float(row[k]) if row.get(k) else np.nan for row in r])
   d=v('index_mcp_object_distance_mm');a=v('finger_width_mm');t=v('t_camera');assert np.isfinite(d[0]) and d[0]>0
   for mm,px in [('index_mcp_object_distance_mm','index_mcp_object_distance'),('finger_width_mm','finger_width')]:
    x=v(px);y=v(mm);valid=np.isfinite(x)&np.isfinite(y);assert np.allclose(y[valid],x[valid]*scale)
   start=meta['onset_row']-meta['start_row'];end=meta['closure_confirmed_row']-meta['start_row'];assert 0<=start<=end<len(r)
   duration=float(t[end]-t[start]);assert duration>0
   normalized=d/d[0];assert normalized[0]==1
   group.append((d,a,normalized,duration))
   audit.append(dict(object_directory=folder,object_side_mm=size,trial=trial.name,input_csv=str(trial/'camera.csv'),source_selection=source,initial_distance_mm=float(d[0]),movement_time_s=duration,total_segment_time_s=float(t[-1]-t[0]),csv_sha256=hashlib.sha256((trial/'camera.csv').read_bytes()).hexdigest()))
  groups.append((folder,size,group))
 max_aperture=max(np.nanmax(a) for _,_,group in groups for _,a,_,_ in group);ylim=float(np.ceil(max_aperture/20)*20)
 max_distance=max(np.nanmax(d) for _,_,group in groups for d,_,_,_ in group);xlim_mm=float(np.ceil(max_distance/50)*50)
 durations=[dur for _,_,g in groups for _,_,_,dur in g];lo=float(np.floor(min(durations)*4)/4);hi=float(np.ceil(max(durations)*4)/4)
 norm=colors.Normalize(vmin=lo,vmax=hi);cmap=plt.get_cmap('viridis')
 for folder,size,group in groups:
  for normalized,subdir in [(True,'initial_distance_normalized')]:
   dest=OUT/subdir;dest.mkdir(parents=True,exist_ok=True)
   fig=plt.figure(figsize=(6.2,6.3),facecolor='white');ax=fig.add_axes([.16,.13,.82,.85])
   for d,a,dnorm,dur in group:ax.plot(dnorm if normalized else d,a,color='#2171b5',lw=.95,alpha=.65)
   ax.set_xlim(0,1.05 if normalized else xlim_mm);ax.set_ylim(0,ylim)
   ax.set_xlabel(r'$d$');ax.set_ylabel(r'$l$ [mm]')
   ax.grid(alpha=.2);ax.set_axisbelow(True)
   if normalized:ax.set_xticks(np.arange(0,1.01,.2))
   assert len(fig.axes)==1 and not ax.get_title() and ax.get_legend() is None
   for ext in ['png','pdf','svg']:fig.savefig(dest/f'{size}mm_overlay.{ext}',dpi=300,facecolor='white')
   plt.close(fig)
 with (OUT/'figure_trial_audit.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(audit[0]));w.writeheader();w.writerows(audit)
 (OUT/'README.txt').write_text('Three separate clean normalized figures: 20, 30, 40 mm; 100 trials each. No title, subtitle, legend or colour bar.\nAxes: d = normalized Euclidean landmark-5-to-object distance divided by its initial value; l = finger aperture in mm. Axis label d is abbreviated as requested; normalization remains applied.\nAll curves use the same blue colour; colour does not encode duration or any other variable.\nPlots use the same Train + Test trials used for the models (one copy of each trial); backup/derived folders excluded. Raw measurements, no smoothing/interpolation or time normalization.\nUser confirmed physical object sides 20/30/40 mm and all scales measured using a 30 mm reference block. Original calibration and CSV fields unchanged.\n',encoding='utf-8')
 print('Saved three clean independent figures (PNG/PDF/SVG):',OUT,'duration colour scale',lo,hi,'shared aperture limit',ylim,flush=True)
if __name__=='__main__':main()
