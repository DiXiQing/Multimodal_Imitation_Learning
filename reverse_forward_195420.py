from pathlib import Path
import csv
import io
import hashlib
import zipfile
import numpy as np
import pandas as pd
from scipy.signal import butter,sosfilt,sosfilt_zi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

source=Path('Data/BackupData/D/TRIAL_Size_3.5cm_20261014_195420')
out=source/'grasp_segments'
audit=out/'sign_correction.csv'
backup=out/'before_sign_correction.zip'
if audit.exists() or backup.exists():
 raise RuntimeError('Sign correction already started or completed; inspect before rerunning')
summary=pd.read_csv(out/'crop_summary.csv')
originals={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in [source/'camera.csv',source/'imu.csv',source/'recorded_video.mp4']}
plans=[]
for trial in summary.trial:
 for name in ['imu.csv','grasp_data.csv','grasp_data_smoothed.csv']:
  p=out/trial/name
  original=p.read_bytes()
  rows=list(csv.reader(io.StringIO(original.decode('utf-8-sig'))))
  targets=['forward_raw','forward_acceleration']
  if name=='imu.csv':targets+=['forward_x','forward_y','forward_z']
  indices=[j for j,col in enumerate(rows[0]) if col in targets]
  assert 'forward_raw' in rows[0]
  changed=[]
  for row in rows[1:]:
   values=row.copy()
   for j in indices:
    assert np.isfinite(float(row[j]))
    values[j]=row[j][1:] if row[j].startswith('-') else '-'+row[j].lstrip('+')
    assert float(values[j])==-float(row[j])
   assert all(values[j]==row[j] for j in range(len(row)) if j not in indices)
   changed.append(values)
  stream=io.StringIO(newline='')
  writer=csv.writer(stream);writer.writerow(rows[0]);writer.writerows(changed)
  plans.append((p,original,stream.getvalue().encode('utf-8'),[rows[0][j] for j in indices]))
with zipfile.ZipFile(backup,'x',compression=zipfile.ZIP_DEFLATED) as z:
 for p,original,_,_ in plans:z.writestr(p.relative_to(out).as_posix(),original)
records=[]
for p,original,updated,columns in plans:
 temporary=p.with_suffix('.tmp.csv');temporary.write_bytes(updated);temporary.replace(p)
 records.append(dict(file=p.relative_to(out).as_posix(),columns=';'.join(columns),factor=-1,before_sha256=hashlib.sha256(original).hexdigest(),after_sha256=hashlib.sha256(updated).hexdigest()))
curves=[]
errors=[]
for row in summary.to_dict('records'):
 p=out/row['trial'];raw=pd.read_csv(p/'grasp_data.csv');sm=pd.read_csv(p/'grasp_data_smoothed.csv')
 imu=pd.read_csv(p/'imu.csv').sort_values('t_imu',kind='stable').drop_duplicates('t_imu',keep='last')
 assert np.array_equal(raw.forward_raw,sm.forward_raw)
 ti=imu.t_imu.to_numpy();tc=raw.t_camera.to_numpy()
 grid=ti[0]+np.arange(int(np.ceil((tc[-1]-ti[0])*50))+1)/50
 u=np.interp(grid,ti,imu.forward_raw);sos=butter(2,3,fs=50,output='sos')
 v=sosfilt(sos,u,zi=sosfilt_zi(sos)*u[0])[0]
 error=float(np.max(np.abs(sm.forward_acceleration-np.interp(tc,grid,v))))
 assert error<2e-5
 errors.append(error);curves.append((row,sm))
for size in [None,3.5,4,4.5]:
 fig,axs=plt.subplots(3,1,figsize=(14,11),sharex=True)
 selected=[(r,d) for r,d in curves if size is None or r['size_cm']==size]
 for i,(r,d) in enumerate(selected):
  x=100*(d.t_camera-d.t_camera.iloc[0])/(d.t_camera.iloc[-1]-d.t_camera.iloc[0])
  color={3.5:plt.cm.Blues,4:plt.cm.Oranges,4.5:plt.cm.Greens}[r['size_cm']](.35+.6*i/max(len(selected)-1,1))
  for ax,col in zip(axs,['finger_width','object_area','forward_acceleration']):ax.plot(x,d[col],color=color,alpha=.65,lw=1)
 for ax,label in zip(axs,['FW (px)','OA (px2)','Signed forward acceleration (m/s2)']):ax.set_ylabel(label);ax.grid(alpha=.2)
 axs[2].axhline(0,color='black',lw=.7);axs[2].set_xlabel('Segment elapsed time (%)')
 fig.suptitle(f'{len(selected)} complete grasps | Blue: 3.5 cm, Orange: 4 cm, Green: 4.5 cm | Forward sign corrected')
 fig.tight_layout();fig.savefig(out/('combined_overlay.png' if size is None else f'overlay_{size:g}cm.png'),dpi=150);plt.close(fig)
assert originals=={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in originals}
pd.DataFrame(records).to_csv(audit,index=False)
with (out/'processing_report.md').open('a',encoding='utf-8') as f:
 f.write('\n\n## 前向符号修正（用户指定）\n\n96 段的 forward_raw 和 forward_acceleration 已逐值乘以 -1。片段 imu.csv 中 forward_raw 及 forward_x/forward_y/forward_z 同步反向，使投影方向一致。此后片段 imu.csv 的这些列是修正值，不再与源文件逐值相同；其他列保持原文本值。ax/ay/az、mag、FW/OA、时间戳、视频、frames 和 camera.csv 未修改。源录制目录下原始文件保持不变。没有均值调整或幅值缩放。原 CSV 已保存于 before_sign_correction.zip，修改哈希见 sign_correction.csv。全部 96 段重建滤波核对通过，最大误差 '+str(max(errors))+' m/s²。合并及三个分尺寸叠加图已更新；recording_overview.png 仍展示未经修正的原始录制。\n')
print('Corrected',len(curves),'trials;',len(records),'CSV files; max filter error',max(errors))
