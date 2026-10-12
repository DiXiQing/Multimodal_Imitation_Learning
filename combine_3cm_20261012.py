from pathlib import Path
import shutil
import pandas as pd
root=Path.cwd();base=root/'Data'/'BackupData'/'D';review=base/'review_3cm_20261012'
out=root/'Data'/'临时数据'/'D12'/'3';out.mkdir(parents=True,exist_ok=True)
d=pd.read_csv(review/'crop_summary.csv')
for name in d.loc[d.status=='valid','experiment']:
    src=base/name;dst=out/name;dst.mkdir(exist_ok=True)
    for p in src.iterdir():
        if p.is_file() and p.suffix in ['.csv','.json','.png']:shutil.copy2(p,dst/p.name)
shutil.copy2(review/'three_signals_overlay.png',out/'three_signals_overlay.png')
shutil.copy2(review/'crop_summary.csv',out/'crop_summary.csv')
assert len(list(out.glob('TRIAL_*/grasp_data_smoothed.csv')))==29
print('Combined folder:',out)
print(d[d.experiment.str[-6:]>'103941'][['experiment','status','reason']].to_string(index=False))
