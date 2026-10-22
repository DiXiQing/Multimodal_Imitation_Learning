from pathlib import Path
import csv,json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]/'Data/TopView/TRIAL_Color_black_Size_3cm_20261022_193242_002419/grasp_segments'
files=sorted(root.glob('*/camera.csv'));summary=[];datasets=[]
fig,axes=plt.subplots(1,2,figsize=(12,4))
for i,p in enumerate(files):
 rows=list(csv.DictReader(p.open(encoding='utf-8-sig')));m=json.loads((p.parent/'segment.json').read_text());t=np.array([float(r['t_camera']) for r in rows]);t-=t[0];a=np.array([float(r['finger_width']) if r['finger_width'] else np.nan for r in rows]);d=np.array([float(r['hand_object_distance']) if r['hand_object_distance'] else np.nan for r in rows]);color=plt.cm.turbo(i/max(1,len(files)-1))
 axes[0].plot(t,a,color=color,alpha=.7,lw=1);axes[1].plot(t,d,color=color,alpha=.7,lw=1)
 frames=np.array([int(r['frame']) for r in rows]);offset=int(m.get('source_start_frame',frames[0]))-int(m['start']);oi=int(np.argmin(abs(frames-(m['onset']+offset))));si=int(np.argmin(abs(frames-(m['stable']+offset))))
 summary.append(dict(trial=p.parent.name,frames=len(rows),duration_s=float(t[-1]),onset_to_stable_s=float(t[si]-t[oi]),peak_aperture_px=float(np.nanmax(a)),invalid_frames=sum(r.get('valid')!='1' for r in rows),note='Initial faster repetition' if i<3 else 'Later repetition'))
 datasets.append((p.parent.name,t,a))
for ax,label in zip(axes,['Fingertip aperture (px)','Hand-object distance (px)']):ax.set(xlabel='Segment time (s)',ylabel=label);ax.grid(alpha=.2)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(root/('grasp_overlay.'+ext),dpi=200)
plt.close(fig)
fig,axes=plt.subplots(int(np.ceil(len(datasets)/5)),5,figsize=(16,2.5*int(np.ceil(len(datasets)/5))),squeeze=False)
for ax,(name,t,a) in zip(axes.flat,datasets):ax.plot(t,a);ax.set_title(name[-6:],fontsize=9);ax.set_xlabel('Time (s)',fontsize=8);ax.grid(alpha=.2)
for ax in list(axes.flat)[len(datasets):]:ax.axis('off')
fig.tight_layout();fig.savefig(root/'aperture_overview.png',dpi=160);plt.close(fig)
with (root/'duration_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
for part in [summary[:3],summary[3:]]:print('n',len(part),'onset-to-stable mean',np.mean([r['onset_to_stable_s'] for r in part]),'duration mean',np.mean([r['duration_s'] for r in part]))
