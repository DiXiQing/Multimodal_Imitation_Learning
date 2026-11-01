"""Rename experiment folders only; preserve recordings, measurements and prior model results."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import csv,json,re,hashlib,shutil
BASE=Path(__file__).resolve().parents[1]
ROOTS=[BASE/'Data/TopView/TRIAL_Color_black_Size_3cm_20261101_212507_486424/grasp_segments',BASE/'Data/1101_D/20mm']
plans=[]
for root in ROOTS:
 root=root.resolve();items=[]
 for parent in (root,root/'review_segments'):
  if not parent.exists():continue
  for p in sorted(parent.iterdir()):
   if not p.is_dir() or not re.fullmatch(r'grasp_\d+',p.name):continue
   with (p/'camera.csv').open(encoding='utf-8-sig',newline='') as f:first=next(csv.DictReader(f))
   name=datetime.fromtimestamp(float(first['t_camera']),timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%S')
   target=p.with_name(name).resolve();source=p.resolve()
   assert source.is_relative_to(root) and target.is_relative_to(root)
   assert source.parent==target.parent and not target.exists()
   inventory={str(f.relative_to(source)):f.stat().st_size for f in source.rglob('*') if f.is_file() and f.name!='segment.json'}
   items.append(dict(old_name=p.name,new_name=name,source=str(source),destination=str(target),relative_source=str(source.relative_to(root)),relative_destination=str(target.relative_to(root)),first_frame=int(first['frame']),first_t_camera=float(first['t_camera']),csv_sha256=hashlib.sha256((p/'camera.csv').read_bytes()).hexdigest(),inventory=inventory))
 assert len({x['new_name'] for x in items})==len(items)
 assert not (root/'timestamp_rename_manifest.json').exists()
 plans.append((root,items))
assert len({x['new_name'] for _,items in plans for x in items})==sum(len(items) for _,items in plans)
for root,items in plans:
 mapping={x['old_name']:x['new_name'] for x in items}
 pattern=re.compile(r'(?<![A-Za-z0-9_])grasp_\d+(?![A-Za-z0-9_])')
 replace=lambda s:pattern.sub(lambda m:mapping.get(m.group(0),m.group(0)),s)
 def update(v):
  if isinstance(v,str):return replace(v)
  if isinstance(v,list):return [update(x) for x in v]
  if isinstance(v,dict):return {replace(k):update(x) for k,x in v.items()}
  return v
 audit=root/'timestamp_rename_metadata_backup';audit.mkdir(exist_ok=False)
 # Prior trained Result directories retain their original trial names as provenance.
 # Original measurement/settings files and videos/images are never edited.
 changed=[]
 files=[p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in ('.json','.csv') and 'Result' not in p.relative_to(root).parts and 'timestamp_rename_metadata_backup' not in p.relative_to(root).parts and p.name not in ('camera.csv','settings.json')]
 for p in files:
  if p.suffix.lower()=='.json':
   before=p.read_text(encoding='utf-8-sig');value=json.loads(before);new=update(value)
   if p.name=='segment.json':new.update(trial_name=mapping.get(p.parent.name,p.parent.name),original_folder_name=p.parent.name)
   if new==value:continue
   relative=p.relative_to(root);dest=audit/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
   p.write_text(json.dumps(new,ensure_ascii=False,indent=2),encoding='utf-8')
  else:
   before=p.read_text(encoding='utf-8-sig');new=replace(before)
   if new==before:continue
   relative=p.relative_to(root);dest=audit/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
   p.write_text(new,encoding='utf-8-sig')
  changed.append(str(relative))
 for item in items:Path(item['source']).rename(item['destination'])
 for item in items:
  target=Path(item['destination'])
  assert target.is_dir() and not Path(item['source']).exists()
  assert hashlib.sha256((target/'camera.csv').read_bytes()).hexdigest()==item['csv_sha256']
  for rel,size in item['inventory'].items():assert (target/rel).is_file() and (target/rel).stat().st_size==size
  # Keep the current relative folder accurate even for previously copied trials.
  segment=target/'segment.json';meta=json.loads(segment.read_text(encoding='utf-8-sig'))
  meta['relative_folder']=item['relative_destination'].replace('\\','/')
  segment.write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
 report=dict(format='YYYYMMDD_HHMMSS',timezone='Asia/Tokyo UTC+09:00',basis='First recorded frame t_camera',renamed_count=len(items),main_count=sum(Path(x['destination']).parent==root for x in items),review_count=sum(Path(x['destination']).parent==root/'review_segments' for x in items),updated_metadata=changed,previous_results_unchanged=True,csv_hashes_unchanged=True,file_names_and_sizes_verified=True,records=[{k:v for k,v in x.items() if k!='inventory'} for x in items])
 (root/'timestamp_rename_manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
 print(root,'renamed',len(items),'main',report['main_count'],'review',report['review_count'],'metadata',len(changed))
