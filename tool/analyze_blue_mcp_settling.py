"""Check residual aperture closure after landmark-5 position settles."""
import os,csv,json
from pathlib import Path
import numpy as np
os.environ['MPLCONFIGDIR']=str(Path(__file__).parent/'.mplconfig')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1]/'Data/TopView/TRIAL_Color_blue_Size_3cm_20261029_122729_581083/grasp_segments'
O=P/'closure_after_mcp_settles';O.mkdir(exist_ok=True)
entries=json.loads((P/'summary.json').read_text());results=[];data={};noise=[]
scale=json.loads((P.parent/'settings.json').read_text(encoding='utf-8-sig'))['pixel_calibration']['mm_per_px']
for e in entries:
 r=list(csv.DictReader(open(P/f"grasp_{e['segment']:03d}"/'camera.csv',encoding='utf-8-sig')))
 v=lambda k:np.array([float(x[k]) for x in r])
 a=v('finger_width_mm');d=v('index_mcp_object_distance_mm');t=v('t_camera')-float(r[0]['t_camera']);xy=np.c_[v('index_mcp_x'),v('index_mcp_y')]*scale
 center=np.median(xy[-10:],axis=0);rad=np.linalg.norm(xy-center,axis=1);dr=np.abs(d-np.median(d[-10:]));target=np.median(a[-10:]);noise.append(float(np.ptp(a[-10:])))
 peak=e['peak_row']-e['start_row'];closure=e['closure_confirmed_row']-e['start_row']
 data[e['segment']]=(t,a,d,rad,r)
 for tol in [2,3,5]:
  for method,err in [('xy_position',rad),('distance_only',dr)]:
   # Strict: every remaining frame stays within the final-hold tolerance.
   cand=[i for i in range(peak,len(r)-10) if np.all(err[i:]<=tol)]
   out=dict(segment=e['segment'],method=method,tolerance_mm=tol,eligible=bool(cand))
   if cand:
    i=cand[0];drop=float(np.median(a[i:i+3])-target)
    out.update(settle_source_frame=int(r[i]['frame']),settle_time_s=float(t[i]),aperture_at_settle_mm=float(np.median(a[i:i+3])),final_aperture_mm=float(target),remaining_closure_mm=drop,closure_confirmation_delay_s=float(t[closure]-t[i]),remaining_xy_max_mm=float(np.max(rad[i:])),remaining_distance_range_mm=float(np.ptp(d[i:])),clear_closure_gt5mm=drop>5,settle_local_index=i)
   results.append(out)
fields=list(dict.fromkeys(k for x in results for k in x))
with (O/'per_segment.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(results)
stats=[]
for tol in [2,3,5]:
 for method in ['xy_position','distance_only']:
  group=[x for x in results if x['tolerance_mm']==tol and x['method']==method and x['eligible']];positive=[x for x in group if x['remaining_closure_mm']>5]
  stats.append(dict(method=method,tolerance_mm=tol,eligible=len(group),clear_gt5mm=len(positive),smaller_gt3mm=sum(x['remaining_closure_mm']>3 for x in group),median_remaining_closure_mm=float(np.median([x['remaining_closure_mm'] for x in group])),positive_segments=[x['segment'] for x in positive]))
main=[x for x in results if x['method']=='xy_position' and x['tolerance_mm']==3 and x['eligible']];examples=sorted(main,key=lambda x:x['remaining_closure_mm'],reverse=True)[:3]
fig,axes=plt.subplots(3,2,figsize=(13,11))
for axrow,x in zip(axes,examples):
 t,a,d,rad,r=data[x['segment']];i=x['settle_local_index'];end=len(t);lo=max(0,i-12)
 ax=axrow[0];ax.plot(t[lo:],a[lo:],marker='.',label='Raw aperture');ax.axhline(x['final_aperture_mm'],color='gray',ls=':');ax.axvline(t[i],color='red',ls='--',label='MCP settles');ax.set_ylabel('Aperture (mm)');ax.set_title(f"Grasp {x['segment']}: remaining closure {x['remaining_closure_mm']:.1f} mm");ax.legend()
 ax=axrow[1];ax.plot(t[lo:],rad[lo:],marker='.',label='MCP offset from final position');ax.axhline(3,color='gray',ls=':');ax.axvline(t[i],color='red',ls='--');ax.set_ylabel('Position offset (mm)');ax.set_ylim(0,max(12,np.max(rad[i:])*1.2));ax.legend()
 for ax in axrow:ax.set_xlabel('Segment actual elapsed time (s)');ax.grid(alpha=.25)
fig.tight_layout();fig.savefig(O/'examples.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(10,5));ax.bar([x['segment'] for x in main],[x['remaining_closure_mm'] for x in main],color=['#db6b40' if x['remaining_closure_mm']>5 else '#477eaa' for x in main]);ax.axhline(5,color='red',ls='--',label='5 mm residual closure threshold');ax.set_xlabel('Grasp segment');ax.set_ylabel('Remaining aperture closure (mm)');ax.set_title('After MCP xy position stays within 3 mm of final hold');ax.legend();fig.tight_layout();fig.savefig(O/'all_segments.png',dpi=170);plt.close(fig)
report=dict(total_segments=len(entries),definitions={'settle':'First post-opening-peak frame after which ALL remaining MCP xy coordinates stay within tolerance of median last-10-frame position; >=11 remaining frames','aperture_drop':'Median first 3 aperture samples at settling minus median last 10 aperture samples','clear_closure':'Aperture drop >5 mm (analysis threshold, not sensor accuracy claim)','closure_delay':'Original segment five-frame closure-confirmation time minus settling time; not exact physical contact time'},sensitivity=stats,hold_aperture_peak_to_peak_mm={'median':float(np.median(noise)),'p95':float(np.percentile(noise,95))},examples=examples,limitations=['2D landmark 5 stability does not establish complete hand or wrist immobility','MediaPipe landmark estimates can move with finger pose and occlusion','1.5 mm/px calibration is approximate at object top height','Raw measurements plotted without smoothing; medians only used for baseline and endpoint summary','Segment endpoints were selected with aperture-based closure rules; this is descriptive timing evidence, not an independent biomechanical experiment'])
(O/'analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
