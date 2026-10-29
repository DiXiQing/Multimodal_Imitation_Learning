"""Finalize visually reviewed trial 42 and refresh plots/metadata; original recording untouched."""
import os,csv,json,hashlib
from pathlib import Path
os.environ['MPLCONFIGDIR']=str(Path(__file__).parent/'.mplconfig')
import cv2,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1]/'Data/TopView/TRIAL_Color_black_Size_3cm_20261029_143159_076439';O=P/'grasp_segments'
def main():
 entries=json.loads((O/'summary.json').read_text())
 with (P/'camera.csv').open(encoding='utf-8-sig',newline='') as f:
  reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
 e=entries[41];e.update(start_row=7354,onset_row=7364,source_start_frame=int(rows[7354]['frame']),frames=e['end_row']-7354+1,duration_s=float(rows[e['end_row']]['t_camera'])-float(rows[7354]['t_camera']))
 e['review_flags']=['Pre-action near-rest only; rapid succession, aperture settling during 10-frame pre-roll; retained for review']
 e['manual_boundary_review']='Start moved to frame 7354 after visual/raw-signal review, excluding previous withdrawal; onset 7364, 10 pre-roll frames; 10 post-closure frames unchanged.'
 dest=O/'grasp_042';seg=rows[e['start_row']:e['end_row']+1]
 with (dest/'camera.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=[*fields,'segment_frame','segment_elapsed_s']);w.writeheader()
  for i,r in enumerate(seg):w.writerow(dict(r,segment_frame=i,segment_elapsed_s=float(r['t_camera'])-float(seg[0]['t_camera'])))
 # Only remove derived images outside the reviewed range, never source images.
 for f in (dest/'frames').glob('frame_*.jpg'):
  frame=int(f.stem.split('_')[-1])
  if frame<e['source_start_frame'] or frame>e['source_end_frame']:f.unlink()
 video=None
 try:
  for r in seg:
   f=dest/'frames'/f"frame_{int(r['frame']):06d}.jpg";source=P/'frames'/f.name
   assert hashlib.sha256(f.read_bytes()).digest()==hashlib.sha256(source.read_bytes()).digest()
   im=cv2.imdecode(np.fromfile(f,np.uint8),1)
   if video is None:
    h,w=im.shape[:2];video=cv2.VideoWriter(str(dest/'grasp_video.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(w,h));assert video.isOpened()
   video.write(im)
 finally:
  if video:video.release()
 cap=cv2.VideoCapture(str(dest/'grasp_video.mp4'));count=0
 while True:
  ok,_=cap.read()
  if not ok:break
  count+=1
 cap.release()
 with (dest/'camera.csv').open(encoding='utf-8-sig',newline='') as f:saved=list(csv.DictReader(f))
 assert len(saved)==len(seg)==count==len(list((dest/'frames').glob('*.jpg')))
 assert all(all(r[k]==s[k] for k in fields) for r,s in zip(seg,saved))
 e['validation']=dict(csv_rows=count,image_frames=count,decoded_video_frames=count,original_fields_preserved=True,image_bytes_preserved=True)
 (dest/'segment.json').write_text(json.dumps(e,indent=2),encoding='utf-8')
 d=np.array([float(r['index_mcp_object_distance_mm']) if r['index_mcp_object_distance_mm'] else np.nan for r in rows]);a=np.array([float(r['finger_width_mm']) if r['finger_width_mm'] else np.nan for r in rows]);t=np.array([float(r['t_camera']) for r in rows])
 def draw(ax,k,group):
  for entry in group:
   s=entry['start_row'];z=entry['end_row']+1;tt=t[s:z]-t[s];x,y=[(d[s:z],a[s:z]),(tt,d[s:z]),(tt,a[s:z])][k];ax.plot(x,y,lw=.8,alpha=.65)
  ax.set_xlabel(['Index MCP distance (mm)','Time (s)','Time (s)'][k]);ax.set_ylabel(['Finger aperture (mm)','Index MCP distance (mm)','Finger aperture (mm)'][k]);ax.grid(alpha=.25)
 fig,ax=plt.subplots(figsize=(8,6));draw(ax,0,[e]);ax.set_title('grasp_042 | distance-aperture | review pre-roll');ax.scatter(d[7354],a[7354],color='green',label='Start');ax.scatter(d[e['end_row']],a[e['end_row']],color='red',marker='s',label='End');ax.legend();fig.tight_layout();fig.savefig(dest/'distance_aperture.png',dpi=160);plt.close(fig)
 for k,name in enumerate(['distance_aperture','distance_time','aperture_time']):
  fig,ax=plt.subplots(figsize=(11,7));draw(ax,k,entries);fig.tight_layout();fig.savefig(O/(name+'.png'),dpi=180);plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(22,6))
 for k,ax in enumerate(axs):draw(ax,k,entries)
 fig.suptitle(f'{len(entries)} grasps | raw measurements | actual elapsed timestamps | mm');fig.tight_layout();fig.savefig(O/'combined_overlays.png',dpi=180);plt.close(fig)
 sheet=np.full((len(entries)*112,800,3),255,np.uint8)
 sample=np.full((4*170,840,3),255,np.uint8)
 for entry in entries:
  n=entry['segment']
  for col,idx in enumerate([entry['start_row'],entry['peak_row'],entry['closure_confirmed_row'] or entry['end_row'],entry['end_row']]):
   im=cv2.imdecode(np.fromfile(P/'frames'/f"frame_{int(rows[idx]['frame']):06d}.jpg",np.uint8),1)
   sheet[(n-1)*112:(n-1)*112+90,col*200:col*200+120]=cv2.resize(im,(120,90));cv2.putText(sheet,f'{n}: {idx}',(col*200,(n-1)*112+105),0,.4,(0,0,0),1)
   if n in [1,42,58,109]:
    y=[1,42,58,109].index(n)*170;x=col*210;sample[y:y+150,x:x+200]=cv2.resize(im,(200,150));cv2.putText(sample,f'{n}: {idx}',(x,y+165),0,.4,(0,0,0),1)
 cv2.imwrite(str(O/'segment_boundaries.jpg'),sheet);cv2.imwrite(str(O/'boundary_sample.jpg'),sample)
 (O/'summary.json').write_text(json.dumps(entries,indent=2),encoding='utf-8');(O/'detection_plan.json').write_text(json.dumps(entries,indent=2),encoding='utf-8')
 with (O/'crop_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for x in entries for k in x)));w.writeheader();w.writerows(entries)
 review=json.loads((O/'detection_review.json').read_text());review['suspicious_segments']=[x for x in entries if x['review_flags']];(O/'detection_review.json').write_text(json.dumps(review,indent=2),encoding='utf-8')
 with (O/'missing_or_suspicious.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(['type','segment','source_start_frame','source_end_frame','note']);w.writerow(['review',42,e['source_start_frame'],e['source_end_frame'],e['review_flags'][0]])
  for m in review['missing_intervals']:w.writerow(['missing_detection','',m['source_start_frame'],m['source_end_frame'],'Original source retained'])
 validation=json.loads((O/'validation.json').read_text())
 for name,h in validation['original_file_sha256'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==h
 validation.update(total_segment_frames=sum(x['frames'] for x in entries),reviewed_boundary_segment=42)
 (O/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
 with (O/'README.txt').open('a',encoding='utf-8') as f:f.write('\nManual review: trial 42 start=7354, onset=7364; near-rest pre-roll has finger settling, retained and flagged. Previous withdrawal excluded.\n')
 print('FINAL VERIFIED:',len(entries),'segments,',sum(x['frames'] for x in entries),'frames; all margins',set(x['pre_action_frames'] for x in entries),set(x['post_closure_frames'] for x in entries),'review',[x['segment'] for x in entries if x['review_flags']])
if __name__=='__main__':main()
