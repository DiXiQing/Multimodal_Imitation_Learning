from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
source = root/'Data/临时数据/D13/3/TRIAL_Size_3cm_20261013_154107_842077'
c = pd.read_csv(source/'camera.csv').sort_values('frame').reset_index(drop=True)
y = c.finger_width.rolling(5,center=True,min_periods=1).median().to_numpy()
peaks,_ = find_peaks(y,prominence=100,distance=70)
records = []
for peak in peaks:
    if y[peak]<160 or peak<40:
        continue
    look = max(0,peak-100)
    low = float(np.min(y[look:peak]))
    below = np.flatnonzero(y[look:peak]<low+.1*(y[peak]-low))
    if not len(below):
        continue
    onset = look+int(below[-1])+1
    start = max(0,onset-25)
    stable = next((j for j in range(peak+8,min(len(y)-8,peak+80))
                   if low+.1*(y[peak]-low)<=np.median(y[j:j+9])<=low+.7*(y[peak]-low)
                   and np.ptp(y[j:j+9])<=10),None)
    if stable is None:
        continue
    end = stable+15
    # Frame inspection: this grasp is released soon after first contact;
    # truncate before the return movement instead of including a false plateau.
    if int(c.frame.iloc[peak]) == 438:
        stable = 457
        end = 462
    gain = float(c.object_area.iloc[stable:end+1].median()/max(c.object_area.iloc[start:onset+1].median(),1))
    if gain<1.25:
        continue
    rec = dict(start=int(c.frame.iloc[start]),end=int(c.frame.iloc[end]),peak=int(c.frame.iloc[peak]),stable=int(c.frame.iloc[stable]),area_gain=gain)
    records.append(rec)
print(json.dumps(records,indent=2))
(source/'candidate_ranges.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
for page in range(0,len(records),7):
    chosen = records[page:page+7]
    fig,axs = plt.subplots(len(chosen),4,figsize=(12,2.4*len(chosen)),squeeze=False)
    for row,rec in enumerate(chosen):
        for ax,key in zip(axs[row],['start','peak','stable','end']):
            k = rec[key]
            path = source/'frames'/f'frame_{k:06d}.jpg'
            img = cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
            ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            ax.set_title(f'Grasp {page+row+1}: {key} {k}')
            ax.axis('off')
    fig.tight_layout()
    fig.savefig(source/f'candidate_frames_{page//7+1}.png',dpi=100)
    plt.close(fig)
