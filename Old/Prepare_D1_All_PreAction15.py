"""Restore five authentic pre-roll rows for all active D1 trials, without changing originals."""
from pathlib import Path
import csv,json,shutil
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'Data/1101_D_PreAction15'
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
 OUT.mkdir(exist_ok=True)
 sources={};audit=[];errors=[]
 for size in [20,30,40]:
  for folder in sorted((ROOT/'Data/1101_D'/f'{size}mm').iterdir()):
   if not (folder/'camera.csv').is_file() or len(folder.name)!=15:continue
   clip=read(folder/'camera.csv');meta=json.loads((folder/'segment.json').read_text(encoding='utf-8-sig'))
   date=folder.name[:8]
   if date not in sources:
    index={}
    for source in (ROOT/'Data/TopView').glob(f'*{date}*/camera.csv'):
     rows=read(source)
     for i,r in enumerate(rows):index[(int(float(r['frame'])),round(float(r['t_camera']),5))]=(source,rows,i)
    sources[date]=index
   match=sources[date].get((int(float(clip[0]['frame'])),round(float(clip[0]['t_camera']),5)))
   if not match or match[2]<5:
    errors.append(dict(size_mm=size,trial=folder.name,reason='Original preceding five rows unavailable'));continue
   source,raw,i=match;pre=raw[i-5:i]
   if any(int(float(b['frame']))-int(float(a['frame']))!=1 or float(b['t_camera'])<=float(a['t_camera']) for a,b in zip(pre+[clip[0]],pre[1:]+[clip[0]])):
    errors.append(dict(size_mm=size,trial=folder.name,reason='Non-contiguous preceding frames'));continue
   rows=pre+clip;keys=list(dict.fromkeys(k for r in rows for k in r))
   rows=[{k:r.get(k,'') for k in keys} for r in rows]
   for j,r in enumerate(rows):
    if 'segment_frame' in keys:r['segment_frame']=j
    if 'segment_elapsed_s' in keys:r['segment_elapsed_s']=float(r['t_camera'])-float(rows[0]['t_camera'])
   dest=OUT/f'{size}mm'/folder.name;dest.mkdir(parents=True,exist_ok=True)
   with (dest/'camera.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
   shutil.copy2(folder/'settings.json',dest/'settings.json')
   original_pre=meta.get('pre_action_frames',int(meta['onset_row'])-int(meta['start_row']))
   meta.update(start_row=int(meta['start_row'])-5,source_start_frame=int(rows[0]['frame']),frames=len(rows),pre_action_frames=original_pre+5,duration_s=float(rows[-1]['t_camera'])-float(rows[0]['t_camera']),restored_preceding_frames=5,restored_source=str(source),original_trial=str(folder),training_csv_copy_only=True)
   meta.pop('validation',None)
   (dest/'segment.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
   audit.append(dict(size_mm=size,trial=folder.name,source=str(source),old_pre_frames=original_pre,new_pre_frames=original_pre+5,added_frames=5))
  print(size,'mm:',sum(r['size_mm']==size for r in audit),'restored',flush=True)
 (OUT/'restoration_manifest.json').write_text(json.dumps(dict(restored=audit,unresolved=errors,originals_modified=False),indent=2),encoding='utf-8')
 print('Restored:',len(audit),'Unresolved:',len(errors),flush=True)
 if errors:print(json.dumps(errors[:10],indent=2),flush=True)
if __name__=='__main__':main()
