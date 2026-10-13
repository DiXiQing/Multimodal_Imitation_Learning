from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
base = root/'Data/临时数据/D13/3/TRIAL_Size_3cm_20261013_154107_842077/grasp_segments'
paths = sorted(base.glob('TRIAL_*/grasp_data_smoothed.csv'))
assert len(paths)==21, 'Expected all 21 segments before selection'
grid = np.linspace(0,100,201)
curves = []
for p in paths:
    d = pd.read_csv(p)
    x = 100*(d.frame-d.frame.iloc[0])/(d.frame.iloc[-1]-d.frame.iloc[0])
    acc = np.linalg.norm(d[['ax','ay','az']],axis=1)
    curves.append(np.stack([np.interp(grid,x,acc),np.interp(grid,x,d.finger_width),np.interp(grid,x,d.object_area)]))
values = np.stack(curves)
reference = np.median(values,axis=0)
# Common physical ranges per signal; no per-trial amplitude normalization.
scales = np.percentile(values,95,axis=(0,2))-np.percentile(values,5,axis=(0,2))
errors = np.sqrt(np.mean((values-reference)**2,axis=2))
normalized = errors/scales
score = normalized@np.array([.5,.25,.25])
order = np.argsort(score)
keep = set(order[:15].tolist())
rows = []
for i,p in enumerate(paths):
    segment = int(p.parent.name.rsplit('segment',1)[1])
    meta = json.loads((p.parent/'crop_metadata.json').read_text())
    rows.append(dict(segment=segment,decision='keep' if i in keep else 'reserve',score=float(score[i]),
                     acceleration_rmse=float(errors[i,0]),fw_rmse_px=float(errors[i,1]),oa_rmse_px2=float(errors[i,2]),
                     frame_start=meta['frame_start'],frame_end=meta['frame_end'],flags=meta.get('flags',[])))
result = pd.DataFrame(rows).sort_values('score')
result.to_csv(base/'selection_15_ranking.csv',index=False,encoding='utf-8-sig')
fig,axs = plt.subplots(6,3,figsize=(15,15),sharex=True)
for row,i in enumerate(order[15:][::-1]):
    for ax,signal,label in zip(axs[row],range(3),['Acceleration (m/s2)','FW (px)','OA (px2)']):
        ax.plot(grid,reference[signal],color='black',lw=2,label='All 21 median')
        ax.plot(grid,values[i,signal],color='tab:red',label=f'Segment {rows[i]["segment"]:02d}')
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axs[row,0].legend(fontsize=8)
for ax in axs[-1]:ax.set_xlabel('Grasp progress (%)')
fig.tight_layout()
fig.savefig(base/'six_removed_comparison.png',dpi=130)
plt.close(fig)
fig,axs = plt.subplots(2,2,figsize=(15,8))
for j,i in enumerate(sorted(keep)):
    for ax,signal in zip([axs[0,0],axs[0,1],axs[1,0]],range(3)):
        ax.plot(grid,values[i,signal],color=plt.cm.tab20(j),lw=1.1,label=f'Grasp {rows[i]["segment"]}')
for ax,label in zip([axs[0,0],axs[0,1],axs[1,0]],['Acceleration magnitude (m/s2)','Finger width (px)','Object area (px2)']):
    ax.set(xlabel='Recorded grasp progress (%)',ylabel=label,xlim=(0,100))
    ax.grid(alpha=.2)
axs[1,1].axis('off')
h,l=axs[0,0].get_legend_handles_labels()
axs[1,1].legend(h,l,ncol=3,loc='center',frameon=False,title='Selected 15 grasps / 3 cm')
fig.tight_layout()
fig.savefig(base/'15_grasps_smooth_overlay.png',dpi=150)
plt.close(fig)
reserve = base/'备用'
reserve.mkdir(exist_ok=True)
for i,p in enumerate(paths):
    if i in keep:continue
    src = p.parent.resolve()
    dst = (reserve/src.name).resolve()
    assert src.parent==base.resolve() and dst.parent==reserve.resolve() and not dst.exists()
    shutil.move(str(src),str(dst))
(base/'selection_15_manifest.json').write_text(json.dumps(dict(method='All 21 median curve reference; common 5-95 percentile ranges; weighted RMSE acceleration 0.5, FW 0.25, OA 0.25',rows=rows),indent=2),encoding='utf-8')
assert len(list(base.glob('TRIAL_*/grasp_data_smoothed.csv')))==15
print(result.to_string(index=False))
print('KEEP', [r['segment'] for r in rows if r['decision']=='keep'])
print('RESERVE', [r['segment'] for r in rows if r['decision']=='reserve'])
