from pathlib import Path
import csv,json,shutil,hashlib,os
import numpy as np,cv2
os.environ['MPLCONFIGDIR']=str(Path(__file__).parent/'.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
B=Path(__file__).resolve().parents[1]/'Data/TopView'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for tag,far in [('215856_066525',185),('220155_064690',140)]:
 S=B/f'TRIAL_Color_blue_Size_3cm_20261031_{tag}';O=S/'grasp_segments';entries=json.loads((O/'summary.json').read_text());settings=json.loads((S/'settings.json').read_text());rows=list(csv.DictReader((S/'camera.csv').open(encoding='utf-8-sig')));fields=list(rows[0]);v=lambda k:np.array([float(r[k]) if r[k] else np.nan for r in rows]);a=v('finger_width');d=v('index_mcp_object_distance');t=v('t_camera');changed=[]
 for x in entries:
  start=x['start_row'];peak=x['peak_row'];end=x['end_row'];lo=entries[x['segment']-2]['near_rows'][1]+1 if x['segment']>1 else 0
  bad=bool(np.nanmax(a[start:min(start+10,peak+1)])>20 or d[start]<far)
  if not bad:continue
  rest=[i for i in range(lo+9,peak) if np.all(a[i-9:i+1]<15) and np.all(d[i-9:i+1]>far)]
  if not rest:continue
  quiet=[i for i in rest if np.ptp(d[i-9:i+1])<=8 and np.ptp(a[i-9:i+1])<=4]
  anchor=quiet[-1] if quiet else rest[-1];newstart=anchor-9;onset=anchor+1
  x['review_flags']=[f for f in x['review_flags'] if f!='No confirmed distant closed rest before action']
  if not quiet:x['review_flags'].append('No quiet 10-frame initial rest (distance range<=8 px, aperture range<=4 px); closest distant closed rest retained')
  if newstart==start:continue
  old=dict(x);x.update(start_row=newstart,onset_row=onset,source_start_frame=int(rows[newstart]['frame']),frames=end-newstart+1,pre_action_frames=10,duration_s=float(t[end]-t[newstart]),boundary_refinement='Removed preceding withdrawal; use last 10-frame distant closed rest before opening, preferring quiet rest')
  dest=O/f"grasp_{x['segment']:03d}";backup=O/'boundary_refinement_backup'/dest.name;backup.parent.mkdir(exist_ok=True)
  # Both absolute paths are verified inside this generated-output directory.
  assert dest.resolve().is_relative_to(O.resolve()) and backup.resolve().is_relative_to(O.resolve())
  shutil.move(str(dest),str(backup));(dest/'frames').mkdir(parents=True)
  segment=rows[newstart:end+1]
  with (dest/'camera.csv').open('w',encoding='utf-8-sig',newline='') as f:
   w=csv.DictWriter(f,fieldnames=[*fields,'segment_frame','segment_elapsed_s']);w.writeheader()
   for i,r in enumerate(segment):w.writerow(dict(r,segment_frame=i,segment_elapsed_s=float(r['t_camera'])-t[newstart]))
  shutil.copy2(S/'settings.json',dest/'settings.json');writer=None
  for r in segment:
   src=S/'frames'/f"frame_{int(r['frame']):06d}.jpg";target=dest/'frames'/src.name;shutil.copy2(src,target);assert digest(src)==digest(target);im=cv2.imdecode(np.fromfile(target,np.uint8),1)
   if writer is None:
    h,w=im.shape[:2];writer=cv2.VideoWriter(str(dest/'grasp_video.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),float(settings['fps']),(w,h));assert writer.isOpened()
   writer.write(im)
  writer.release();cap=cv2.VideoCapture(str(dest/'grasp_video.mp4'));count=0
  while cap.read()[0]:count+=1
  cap.release();saved=list(csv.DictReader((dest/'camera.csv').open(encoding='utf-8-sig')));assert len(saved)==len(segment)==count==len(list((dest/'frames').glob('*.jpg')));assert all(all(r[k]==q[k] for k in fields) for r,q in zip(segment,saved))
  x['validation']=dict(csv_rows=count,image_frames=count,decoded_video_frames=count,original_fields_preserved=True,image_bytes_preserved=True);(dest/'segment.json').write_text(json.dumps(x,indent=2),encoding='utf-8');changed.append(dict(segment=x['segment'],old_start_row=old['start_row'],new_start_row=newstart,flags=x['review_flags']))
 (O/'summary.json').write_text(json.dumps(entries,indent=2),encoding='utf-8')
 allfields=list(dict.fromkeys(k for x in entries for k in x))
 with (O/'crop_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=allfields);w.writeheader();w.writerows(entries)
 review=json.loads((O/'detection_review.json').read_text());review['suspicious_segments']=[x for x in entries if x['review_flags']];review['boundary_refinements']=changed;(O/'detection_review.json').write_text(json.dumps(review,indent=2),encoding='utf-8')
 dd=v('index_mcp_object_distance_mm');aa=v('finger_width_mm');spec=[('distance_aperture','Index MCP distance (mm)','Aperture (mm)'),('distance_time','Time (s)','Index MCP distance (mm)'),('aperture_time','Time (s)','Aperture (mm)')]
 def draw(ax,k):
  for x in entries:
   s=x['start_row'];e=x['end_row']+1;tt=t[s:e]-t[s];xx,yy=[(dd[s:e],aa[s:e]),(tt,dd[s:e]),(tt,aa[s:e])][k];ax.plot(xx,yy,lw=.8,alpha=.65)
  ax.set_xlabel(spec[k][1]);ax.set_ylabel(spec[k][2]);ax.grid(alpha=.25)
 for k,(name,_,__) in enumerate(spec):
  fig,ax=plt.subplots(figsize=(11,7));draw(ax,k);fig.tight_layout();fig.savefig(O/(name+'.png'),dpi=180);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(22,6))
 for k,ax in enumerate(axes):draw(ax,k)
 fig.suptitle(f'{len(entries)} grasps | raw measurements | actual elapsed timestamps | mm');fig.tight_layout();fig.savefig(O/'combined_overlays.png',dpi=180);plt.close(fig)
 sheet=np.full((len(entries)*112,800,3),255,np.uint8)
 for j,x in enumerate(entries):
  for c,idx in enumerate([x['start_row'],x['peak_row'],x['closure_confirmed_row'] or x['end_row'],x['end_row']]):
   im=cv2.imdecode(np.fromfile(S/'frames'/f"frame_{int(rows[idx]['frame']):06d}.jpg",np.uint8),1);sheet[j*112:j*112+90,c*200:c*200+120]=cv2.resize(im,(120,90));cv2.putText(sheet,f"{j+1}: {rows[idx]['frame']}",(c*200,j*112+105),0,.4,(0,0,0),1)
 cv2.imwrite(str(O/'segment_boundaries.jpg'),sheet)
 validation=json.loads((O/'validation.json').read_text());assert all(digest(S/name)==h for name,h in validation['original_file_sha256'].items());validation['boundary_refinement_complete']=True;validation['active_segments_only']='grasp_###; boundary_refinement_backup contains superseded generated exports, not extra trials';(O/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
 with (O/'README.txt').open('a',encoding='utf-8') as f:f.write('\nBoundary refinement: removed preceding withdrawal in selected starts. Current exports are grasp_###; superseded generated exports preserved in boundary_refinement_backup. Prefer last distant closed 10-frame rest with distance range <=8px and aperture range <=4px; flag fallback rests. See detection_review.json.\n')
 print(tag,'segments',len(entries),'refined',changed,'review',[(x['segment'],x['review_flags']) for x in entries if x['review_flags']],flush=True)
