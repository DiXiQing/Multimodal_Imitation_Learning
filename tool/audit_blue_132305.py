"""Supplemental pre-action stillness review; never removes candidates."""
import json,csv,numpy as np
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'Data/TopView/TRIAL_Color_blue_Size_3cm_20261029_132305_270675/grasp_segments'
def main():
 entries=json.loads((P/'summary.json').read_text());audit=[]
 for e in entries:
  f=P/f"grasp_{e['segment']:03d}"/'camera.csv'
  with f.open(encoding='utf-8-sig',newline='') as stream:r=list(csv.DictReader(stream))
  n=e['pre_action_frames'];d=np.array([float(x['index_mcp_object_distance']) for x in r[:n]]);a=np.array([float(x['finger_width']) for x in r[:n]])
  drift=float(np.ptp(d));ap=float(np.ptp(a));item=dict(segment=e['segment'],pre_distance_range_px=drift,pre_aperture_range_px=ap)
  if drift>6 or ap>4:
   flag=f'Pre-action stillness review: distance range {drift:.3f} px, aperture range {ap:.3f} px (gates 6/4 px)'
   if flag not in e['review_flags']:e['review_flags'].append(flag)
   (f.parent/'segment.json').write_text(json.dumps(e,indent=2),encoding='utf-8')
  item['review_flags']=e['review_flags'];audit.append(item)
 (P/'summary.json').write_text(json.dumps(entries,indent=2),encoding='utf-8')
 with (P/'crop_summary.csv').open('w',newline='',encoding='utf-8-sig') as stream:
  w=csv.DictWriter(stream,fieldnames=list(entries[0]));w.writeheader();w.writerows(entries)
 f=P/'detection_review.json';review=json.loads(f.read_text());review['suspicious_segments']=[e for e in entries if e['review_flags']];review['supplemental_rule']='Pre-action 10-frame raw distance range >6 px or aperture range >4 px: review only; keep segment.';f.write_text(json.dumps(review,indent=2),encoding='utf-8')
 (P/'pre_action_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
 print('segments',len(entries),'frames',sum(e['frames'] for e in entries),'flagged',[e['segment'] for e in entries if e['review_flags']]);assert all(e['pre_action_frames']==10 and e['post_closure_frames']==10 for e in entries)
if __name__=='__main__':main()
