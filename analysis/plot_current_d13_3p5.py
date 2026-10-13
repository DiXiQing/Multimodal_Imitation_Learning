from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
base=root/'Data'/'临时数据'/'D13'/'3.5'
trials=sorted(t for t in base.glob('TRIAL_*') if (t/'grasp_data_smoothed.csv').exists())
assert trials,'No processed trials found'
fig,axs=plt.subplots(2,2,figsize=(16,8));axes=[axs[0,0],axs[0,1],axs[1,0]]
for i,t in enumerate(trials):
    d=pd.read_csv(t/'grasp_data_smoothed.csv');x=100*(d.frame-d.frame.iloc[0])/(d.frame.iloc[-1]-d.frame.iloc[0])
    ys=[d.finger_width,d.object_area,np.linalg.norm(d[['ax','ay','az']],axis=1)]
    for ax,y in zip(axes,ys):ax.plot(x,y,lw=1.2,color=plt.cm.tab20(i/max(1,len(trials)-1)),label=t.name[-6:])
for ax,label in zip(axes,['Finger width (px)','Object area (px²)','Acceleration magnitude (m/s²)']):ax.set(xlabel='Recorded trial progress (%)',ylabel=label,xlim=(0,100));ax.grid(alpha=.2)
axs[1,1].axis('off');h,l=axes[0].get_legend_handles_labels();axs[1,1].legend(h,l,ncol=3,loc='center',frameon=False,title=f'3.5 cm — {len(trials)} trials')
fig.tight_layout();out=base/'3.5cm_smooth_combined_acceleration_overlay.png';fig.savefig(out,dpi=150);plt.close(fig)
print('Trials:',len(trials));print(out)


