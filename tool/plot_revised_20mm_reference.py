"""Plot user-corrected aperture reference against unchanged saved predictions."""
import os,csv,json,hashlib
from pathlib import Path
os.environ['MPLCONFIGDIR']=str(Path(__file__).parent/'.mplconfig')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1]/'Data/1029_D1';trial=B/'20mm_red/Train/grasp_051';run=B/'20mm_red/Result/l0_normalized_function_mlp5_lstmfull_400ep_20261029_162159_416592';source=run/'Train/predictions/grasp_051.csv'
original_hash=hashlib.sha256(source.read_bytes()).hexdigest()
with (trial/'camera.csv').open(encoding='utf-8-sig',newline='') as f:raw={int(x['frame']):x for x in csv.DictReader(f)}
with source.open(encoding='utf-8-sig',newline='') as f:r=list(csv.DictReader(f))
t=np.array([float(x['time_s']) for x in r]);y=np.array([float(raw[int(x['source_frame'])]['finger_width_mm']) for x in r]);common=np.array([x['common_valid']=='1' for x in r])
fig,ax=plt.subplots(figsize=(4.6,5.6),facecolor='white');fig.subplots_adjust(left=.18,right=.97,bottom=.12,top=.98)
ax.plot(t,y,color='black',lw=1.7,label='Reference (edited)')
for key,label,color in zip(['function_mm','mlp_5f_mm','lstm_full_mm'],['Function','MLP 5f','LSTM Full'],['#1f77b4','#ff7f0e','#2ca02c']):
 p=np.array([float(x[key]) if x[key] else np.nan for x in r]);p[~common]=np.nan;ax.plot(t,p,color=color,lw=1.5,label=label)
ax.set_xlabel('Time (s)',fontsize=12);ax.set_ylabel('Finger aperture (mm)',fontsize=12);ax.grid(alpha=.2);ax.legend(loc='upper right',fontsize=7.5,handlelength=1.5,labelspacing=.3,borderpad=.35);ax.tick_params(labelsize=10)
out=B/'report_prediction_examples/closing_phase_selected';name='20mm_Train_grasp_051_revised_reference'
for ext in ['png','pdf','svg']:fig.savefig(out/f'{name}.{ext}',dpi=300,facecolor='white')
plt.close(fig)
(out/f'{name}.json').write_text(json.dumps(dict(reference_csv=str((trial/'camera.csv').resolve()),reference_status='User-manually-edited px aperture, synchronized to mm using settings calibration; not unmodified raw measurement',predictions_csv=str(source.resolve()),predictions_status='Unchanged saved predictions; models not retrained',metrics_status='Original run metrics unchanged and predate reference correction',source_frame_alignment=True),indent=2),encoding='utf-8')
assert hashlib.sha256(source.read_bytes()).hexdigest()==original_hash
print(out/f'{name}.png')
