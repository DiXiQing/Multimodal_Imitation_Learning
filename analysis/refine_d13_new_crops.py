from pathlib import Path
import sys,json,shutil,datetime,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from Analyze import calculate_3d_speed
from Smooth import smooth_trial
base=root/'Data'/'临时数据'/'D13';review=base/'review_new_3.5_4'
backup=root/'analysis'/('D13_crop_refinement_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));backup.mkdir()
for suffix in ['131624','131652']:
    t=next(base.glob('TRIAL_*_'+suffix));meta=json.loads((t/'crop_metadata.json').read_text())
    c=pd.read_csv(t/'camera.csv');y=c.finger_width.rolling(5,center=True,min_periods=1).median().to_numpy();peak=int(np.flatnonzero(c.frame.to_numpy()==meta['peak_frame'])[0])
    local=max(0,peak-40);baseline=np.median(y[local:max(local+1,peak-15)])
    below=np.flatnonzero(y[local:peak]<baseline+.1*(y[peak]-baseline));onset=local+int(below[-1])+1
    start=max(0,onset-25);end=meta['frame_end'];d=c.loc[c.frame.between(start,end)].copy()
    b=backup/t.name;b.mkdir()
    for name in ['grasp_data.csv','grasp_data_smoothed.csv','crop_metadata.json','smoothing_preview.png']:shutil.copy2(t/name,b/name)
    im=pd.read_csv(t/'imu.csv').sort_values('t_imu').drop_duplicates('t_imu',keep='last');tc=d.t_camera.to_numpy();im=im.loc[im.t_imu.between(tc[0],tc[-1])];ti=im.t_imu.to_numpy();a=im[['ax','ay','az']].to_numpy();_,_,_,speed=calculate_3d_speed(ti,a)
    for j,col in enumerate(['ax','ay','az']):d[col]=np.interp(tc,ti,a[:,j])
    d['mag']=np.linalg.norm(d[['ax','ay','az']],axis=1);d['velocity']=np.interp(tc,ti,speed)
    smooth=smooth_trial(d);smooth['mag']=np.linalg.norm(smooth[['ax','ay','az']],axis=1)
    d.to_csv(t/'grasp_data.csv',index=False);smooth.to_csv(t/'grasp_data_smoothed.csv',index=False,float_format='%.9g')
    fig,axs=plt.subplots(3,1,figsize=(11,8),sharex=True)
    for ax,col in zip(axs,['finger_width','object_area','mag']):ax.plot(d.frame,d[col],alpha=.4,label='Raw');ax.plot(smooth.frame,smooth[col],label='Smoothed');ax.set_ylabel(col);ax.grid(alpha=.2)
    axs[0].legend();axs[-1].set_xlabel('Frame');fig.tight_layout();fig.savefig(t/'smoothing_preview.png',dpi=140);plt.close(fig)
    meta.update(frame_start=int(start),rows=len(d),initial_acceleration=float(smooth.mag.iloc[:5].mean()),final_acceleration=float(smooth.mag.iloc[-5:].mean()),acc_peak=float(smooth.mag.max()),fw_peak=float(smooth.finger_width.max()),boundary_refinement='Local pre-grasp baseline avoids earlier preparation movements')
    meta['flags']=[f for f in meta['flags'] if 'Long crop' not in f]
    (t/'crop_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print(suffix,start,end,len(d))
summary=pd.read_csv(review/'crop_and_group_summary.csv');rows=[]
for t in sorted(base.glob('TRIAL_*')):
    if not (t/'crop_metadata.json').exists():
        rows.extend(summary.loc[summary.trial==t.name].to_dict('records'));continue
    meta=json.loads((t/'crop_metadata.json').read_text());d=pd.read_csv(t/'grasp_data.csv')
    zero=int((d.finger_width<=0).sum());meta['fw_zero_frames']=zero
    if zero:meta['flags'].append(f'FW detection missing in {zero} cropped frames; not repaired')
    if zero>10:meta['status']='cropped_review_required'
    (t/'crop_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    for name,h in meta['source_sha256'].items():assert hashlib.sha256((t/name).read_bytes()).hexdigest()==h
    rows.append(meta)
pd.DataFrame(rows).to_csv(review/'crop_and_group_summary.csv',index=False,encoding='utf-8-sig');(review/'grouping_manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
for size in ['3.5cm','4cm']:
    trials=sorted(base.glob('TRIAL_Size_'+size+'_*'));trials=[t for t in trials if (t/'grasp_data_smoothed.csv').exists()]
    fig,axs=plt.subplots(2,2,figsize=(16,8));sas=[axs[0,0],axs[0,1],axs[1,0]]
    for i,t in enumerate(trials):
        d=pd.read_csv(t/'grasp_data_smoothed.csv');x=100*(d.frame-d.frame.iloc[0])/(d.frame.iloc[-1]-d.frame.iloc[0])
        for ax,col in zip(sas,['finger_width','object_area','mag']):ax.plot(x,d[col],lw=1.2,color=plt.cm.tab20(i/max(1,len(trials)-1)),label=t.name[-6:]+(' [check]' if t.name.endswith('130447') else ''))
    for ax,label in zip(sas,['Finger width (px)','Object area (px²)','Acceleration magnitude (m/s²)']):ax.set(xlabel='Recorded trial progress (%)',ylabel=label,xlim=(0,100));ax.grid(alpha=.2)
    axs[1,1].axis('off');h,l=sas[0].get_legend_handles_labels();axs[1,1].legend(h,l,ncol=3,loc='center',frameon=False,title=f'{size}: {len(trials)} trials');fig.tight_layout();fig.savefig(review/(size+'_overlay.png'),dpi=150);plt.close(fig)
# Replace stale boundary review pages with the final boundaries.
valid=[r for r in rows if r.get('frame_start') is not None and pd.notna(r.get('frame_start'))]
for offset in range(0,len(valid),6):
    items=valid[offset:offset+6];fig,axs=plt.subplots(len(items),1,figsize=(15,2.7*len(items)),squeeze=False)
    for ax,r in zip(axs[:,0],items):
        c=pd.read_csv(base/r['trial']/'camera.csv');ax.plot(c.frame,c.finger_width.rolling(5,center=True,min_periods=1).median());ax.axvspan(r['frame_start'],r['frame_end'],color='green',alpha=.2);ax.set_title(r['trial'][-6:]);ax.set_ylabel('FW (px)');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(review/f'crop_review_{offset//6+1}.png',dpi=120);plt.close(fig)
print(pd.DataFrame(rows).groupby(['group','status']).size().to_string())
