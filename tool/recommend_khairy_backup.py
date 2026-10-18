from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from train_window_mlp import predict_trial,regression_metrics
BASE=ROOT/'Data/Exprement Data/Khairy'
OUT=ROOT/'Report/khairy_quality_review'
RUN=BASE/'Result/generalization_20261016_112916_799339'
audit=pd.read_csv(OUT/'all_trial_quality.csv')
names={
'TRIAL_Size_3cm_20261016_105127':'OA drops to 67% of baseline while the hand occludes the object',
'TRIAL_Size_3cm_20261016_105243':'OA drops to 43% of baseline while the hand occludes the object',
'TRIAL_Size_3cm_20261016_105247':'OA drops to 52% of baseline while the hand occludes the object',
'TRIAL_Size_3.5cm_20261016_104658':'OA drops to 53% of baseline while the hand occludes the object',
'TRIAL_Size_3.5cm_20261016_104730':'OA drops to 36% of baseline while the hand occludes the object',
'TRIAL_Size_4cm_20261016_104458':'OA drops to 64% of baseline while the hand occludes the object',
'TRIAL_Size_4cm_20261016_104507':'OA drops to 71% of baseline while the hand occludes the object',
'TRIAL_Size_4cm_20261016_104529':'OA drops to 58% of baseline while the hand occludes the object',
'TRIAL_Size_4cm_20261016_104554':'OA drops to 65% of baseline while the hand occludes the object',
'TRIAL_Size_3cm_20261016_105132':'Only 36 frames; onset already underway at segment start; recrop from continuous source',
}
recommended=audit.loc[audit.trial.isin(names)].copy()
recommended['reason']=recommended.trial.map(names)
recommended['recommendation']=np.where(recommended.trial.str.endswith('105132'),'recrop_before_use','backup_or_repair_object_detection')
recommended.to_csv(OUT/'backup_recommendations.csv',index=False,encoding='utf-8-sig')
records=[]
for model in ['window_area_only','window_area_acceleration']:
    path=RUN/'models'/f'{model}.pth'
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    names_train=set(checkpoint['train_trials'])
    for row in audit.itertuples():
        trial=BASE/f'{row.size_cm:g}'/row.trial
        d=pd.read_csv(trial/'grasp_data_smoothed.csv')
        target,prediction,_=predict_trial(path,d,torch.device('cpu'))
        rmse,mae,r2=regression_metrics(target,prediction)
        records.append(dict(trial=row.trial,model=model,split='train' if row.trial in names_train else row.split,rmse=rmse,mae=mae,r2=r2,quality_flag=row.trial in names))
metrics=pd.DataFrame(records);metrics.to_csv(OUT/'model_fit_diagnostic.csv',index=False)
print(metrics.groupby(['model','split'])[['rmse','mae','r2']].mean().to_string())
print('Recommendations',len(recommended))
print(recommended[['trial','split','recommendation']].to_string(index=False))
