from pathlib import Path
import csv,sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'Program'))
from Replace_D1_Two_Long_Examples import replay,read,base,RESULT,OUT

def phase_score(rows):
 t=np.array([float(r['time_s']) for r in rows]);y=np.array([float(r['actual_mm']) for r in rows]);p=np.array([float(r['lstm_full_mm']) for r in rows]);baseline=float(np.median(y[:8]));peak=int(np.argmax(y));amplitude=y[peak]-baseline
 start=np.flatnonzero(y[:peak+1]>=baseline+max(5,.1*amplitude))
 end=np.flatnonzero(y[:peak+1]>=baseline+.65*amplitude)
 if not len(start) or not len(end):return None
 a=int(start[0]);b=int(end[0]);mask=(np.arange(len(y))>=max(0,a-2))&(np.arange(len(y))<=b)&np.isfinite(p)
 if mask.sum()<3:return None
 onset_pred=np.flatnonzero((p>=baseline+max(5,.1*amplitude)) & (np.arange(len(y))<=peak))
 if not len(onset_pred):return None
 lag=abs(t[onset_pred[0]]-t[a]);err=float(np.mean(np.abs(y[mask]-p[mask])))
 return err,float(lag),float(t[a]),float(t[b])
def main():
 torch.set_num_threads(2)
 selection=read(OUT/'D1_updated_four_examples_sources.csv');oldmetrics=read(RESULT/'oof_per_trial_metrics.csv');conditions=read(ROOT/'Report/Two_Participant_Comparison/Replacement_3p0s/current600_conditions.csv');used={r['trial'] for r in selection};details=[];candidate_audit=[]
 for size,group,lo,hi in [(30,'Medium',1.5,2.25),(20,'Long',2.1,3.),(30,'Long',2.1,3.),(40,'Long',2.1,3.)]:
  options=[]
  for m in oldmetrics:
   if m['model']!='LSTM Full' or int(m['size_cm'])*10!=size or m['trial'] in used:continue
   rec=next((r for r in conditions if int(r['size_mm'])==size and r['trial']==m['trial']),None)
   if not rec or not lo<=float(rec['action_duration_s'])<=hi:continue
   folder=ROOT/'Data/1101_D'/f'{size}mm'/m['trial'];a=read(folder/'camera.csv');b=read(ROOT/'Data/1101_D_PreAction15'/f'{size}mm'/m['trial']/'camera.csv')
   if float(a[-1]['t_camera'])-float(a[0]['t_camera'])>3 or len(b)<len(a):continue
   b=b[-len(a):]
   if any(not np.allclose([float(x[k]) for x in a],[float(x[k]) for x in b],equal_nan=True,rtol=0,atol=1e-6) for k in ['t_camera','finger_width_mm','index_mcp_object_distance_mm']):continue
   rows,scores=replay(folder,size,int(m['fold']));phase=phase_score(rows)
   if phase is None:continue
   err,lag,t0,t1=phase
   qualifies=scores['lstm_full_mm']<min(scores['function_mm'],scores['mlp_5f_mm'])
   candidate_audit.append(dict(size_mm=size,group=group,trial=m['trial'],rise_mae_mm=err,onset_error_s=lag,lstm_mae_mm=scores['lstm_full_mm'],lstm_lowest_mae=qualifies))
   options.append((err+10*lag+.1*scores['lstm_full_mm'],folder,rows,scores,phase,qualifies,int(m['fold'])))
  good=options if size==20 else [r for r in options if r[5]]
  if not good:good=options
  if not good:raise ValueError(f'No candidates {size} {group}')
  _,folder,rows,scores,phase,qualifies,fold=min(good,key=lambda r:r[0]);used.add(folder.name)
  source=OUT/'predictions'/f'{size}mm'/f'{folder.name}_rise_match.csv';base.write_csv(source,rows)
  target=next(r for r in selection if int(r['size_mm'])==size and r['duration_group']==group)
  oldphase=phase_score(read(Path(target['source'])))
  target.update(trial=folder.name,source=str(source),status='Selected for low early-rise error and onset discrepancy; unchanged current trial, held-out fold replay; not representative sampling')
  detail=dict(size_mm=size,group=group,trial=folder.name,rise_mae_mm=phase[0],onset_error_s=phase[1],old_rise_mae_mm=oldphase[0] if oldphase else None,old_onset_error_s=oldphase[1] if oldphase else None,lstm_lowest_mae=qualifies,**scores);details.append(detail);print(detail,flush=True)
 fig,axes=plt.subplots(3,3,figsize=(16,8),dpi=220);handles=[]
 for c,size in enumerate([20,30,40]):
  for row,group in enumerate(['Short','Medium','Long']):
   rec=next(r for r in selection if int(r['size_mm'])==size and r['duration_group']==group);data=read(Path(rec['source']));ax=axes[row,c]
   for key,label,color in [('actual_mm','Measured','#222222'),('function_mm','Function','#2786bd'),('mlp_5f_mm','MLP 5f','#f09527'),('lstm_full_mm','LSTM Full','#389b51')]:
    y=np.array([float(r[key]) for r in data])
    if key!='actual_mm' and 'common_valid' in data[0]:y[np.array([r['common_valid']!='1' for r in data])]=np.nan
    line,=ax.plot([float(r['time_s']) for r in data],y,label=label,color=color,lw=1.4)
    if row==0 and c==0:handles.append(line)
   ax.set_xlim(0,3);ax.set_ylim(0,125);ax.set_xticks(np.arange(0,3.01,.5));ax.set_yticks(np.arange(0,126,25));ax.set_title(f'{size} mm',fontsize=16,pad=9);ax.set_xlabel('Time (s)',fontsize=12);ax.tick_params(labelsize=11);ax.grid(alpha=.22);ax.set_axisbelow(True)
   if c==0:ax.set_ylabel('Finger aperture d(mm)',fontsize=12)
   else:ax.tick_params(labelleft=False)
 fig.legend(handles=handles,loc='upper right',bbox_to_anchor=(.993,1.005),ncol=4,frameon=False,fontsize=13);fig.tight_layout(rect=(0,0,1,.955),h_pad=1.4,w_pad=1.4)
 for ext in ['png','svg','pdf']:fig.savefig(OUT/f'D1_rise_matched_examples_time3s.{ext}')
 base.write_csv(OUT/'D1_rise_matched_sources.csv',selection);base.write_csv(OUT/'rise_match_replacement_metrics.csv',details);base.write_csv(OUT/'rise_match_candidate_audit.csv',candidate_audit)
if __name__=='__main__':main()

