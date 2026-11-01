from pathlib import Path
import csv,json,shutil,hashlib,random,time
B=Path(__file__).resolve().parents[1];D=B/'Data/1031_D/30mm_mixed_215856_220155'
if D.exists():raise RuntimeError(f'Destination exists; refusing overwrite: {D}')
D.mkdir();records=[];removed=[];n=0
for tag in ['215856_066525','220155_064690']:
 S=B/f'Data/TopView/TRIAL_Color_blue_Size_3cm_20261031_{tag}/grasp_segments'
 entries=json.loads((S/'summary.json').read_text())
 for x in entries:
  n+=1;src=S/f"grasp_{x['segment']:03d}";name=f'grasp_{n:03d}';dest=D/name
  shutil.copytree(src,dest)
  if x['review_flags']:
   # Delete only this newly copied abnormal segment, never original recordings.
   target=dest.resolve();assert target.parent==D.resolve() and target.name==name and target.is_dir()
   shutil.rmtree(target)
   removed.append(dict(merged_trial=name,source_recording=tag,source_trial=src.name,reason=x['review_flags']));continue
  for f in src.rglob('*'):
   if f.is_file():
    g=dest/f.relative_to(src);assert hashlib.sha256(f.read_bytes()).digest()==hashlib.sha256(g.read_bytes()).digest(),str(g)
  with (dest/'camera.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
  images=len(list((dest/'frames').glob('*.jpg')));assert len(rows)==images==x['validation']['decoded_video_frames']==x['frames']
  assert not x['review_flags']
  records.append(dict(trial=name,source_recording=tag,source_trial=src.name,source_path=str(src),frames=len(rows),csv_sha256=hashlib.sha256((dest/'camera.csv').read_bytes()).hexdigest()))
 print('Copied',tag,'retained',len(records),'removed',len(removed),flush=True)
assert len(records)==113 and len(removed)==6
# Global whole-trial shuffle, same fixed seed as the training scripts.
order=list(range(len(records)));random.Random(20261029).shuffle(order);trainset=set(order[:int(len(order)*.75)])
for i,r in enumerate(records):
 split='Train' if i in trainset else 'Test';r['split']=split;(D/split).mkdir(exist_ok=True);shutil.move(str(D/r['trial']),str(D/split/r['trial']))
 for f in (D/split/r['trial']).glob('camera.csv'):assert hashlib.sha256(f.read_bytes()).hexdigest()==r['csv_sha256']
assert len(list((D/'Train').glob('grasp_*')))==84 and len(list((D/'Test').glob('grasp_*')))==29
manifest=dict(source_recordings=['20261031_215856_066525','20261031_220155_064690'],source_total=119,kept=113,deleted_from_mixed_dataset=removed,seed=20261029,strategy='Whole-experiment global random 75/25 across both recordings; original recording files preserved',train_count=84,test_count=29,copy_verified='Every source file byte-identical after copying; CSV/images and previously decoded video counts match; source video bytes verified',records=records)
(D/'merge_and_split_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
for filename,data in [('dataset_manifest.csv',records),('deleted_abnormal_segments.csv',removed)]:
 with (D/filename).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
(D/'README.txt').write_text('Mixed 30mm blue dataset, 113 retained trials from two recordings. Six flagged abnormal copied segments deleted from this dataset. Source recordings untouched. Whole-trial random 84 Train / 29 Test; seed 20261029. Each trial occurs in exactly one split. Manifest maps renamed trials to source recordings.\n',encoding='utf-8')
print('READY',D,'Train84 Test29',flush=True)
