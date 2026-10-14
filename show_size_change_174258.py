import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.append(r'C:\Users\Di\AppData\Roaming\Python\Python312\site-packages')
import cv2
from pathlib import Path
s=Path(r'C:\MineApp\Code\Multimodal_Imitation_Learning\Data\BackupData\D\TRIAL_Size_2.5cm_20261014_174258')
fig,axs=plt.subplots(1,4,figsize=(14,4))
for ax,k,title in zip(axs,[6630,6670,6890,6991],['2.5 cm: before replacement','17:46:42: removing old object','17:46:49: placing new object','3 cm: first segment, 17:46:52']):
 im=cv2.imdecode(np.fromfile(s/'frames'/f'frame_{k:06d}.jpg',np.uint8),1);ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));ax.set_title(title+'\nFrame '+str(k),fontsize=10);ax.axis('off')
fig.tight_layout();fig.savefig(s/'grasp_segments'/'size_change_evidence.png',dpi=150)
