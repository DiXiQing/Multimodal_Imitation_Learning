from pathlib import Path
import pandas as pd
run=Path.cwd()/'generalization_20261011_185104_437403'
folder=(run/'generalization_predictions').resolve()
assert folder.parent==run.resolve()
d=pd.read_csv(run/'generalization_trial_metrics.csv')
a=d[d.model=='window_area_acceleration'].sort_values(['object_size_cm','rmse_px','trial'])
keep=a.groupby('object_size_cm',group_keys=False).head(5)
assert len(keep)==15 and set(keep.object_size_cm)=={2.5,3.5,4.5}
files=list(folder.glob('*.png'))
assert all(p.stem in set(a.trial) and p.resolve().parent==folder for p in files)
assert set(keep.trial).issubset({p.stem for p in files})
manifest=a.copy();manifest['retained']=manifest.trial.isin(keep.trial)
manifest.to_csv(run/'generalization_best5_selection.csv',index=False,encoding='utf-8-sig')
removed=0
for p in files:
    if p.stem not in set(keep.trial):p.unlink();removed+=1
assert len(list(folder.glob('*.png')))==15
print(keep[['object_size_cm','trial','rmse_px']].to_string(index=False));print('Removed images:',removed)
