"""Read-only diagnosis of initial MCP distance reversal; causal EMA preview only."""
from pathlib import Path
import argparse,csv,json,hashlib,os
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).parent/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parents[1]

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--source',type=Path,default=BASE/'Data/1101_D/20mm/20261101_195814')
 parser.add_argument('--tau',type=float,default=.08,help='Preview EMA time constant in seconds; not a learned correction')
 parser.add_argument('--initial-seconds',type=float,default=.4)
 args=parser.parse_args()
 if args.tau<=0 or args.initial_seconds<=0:parser.error('Positive time constants/ranges required')
 source=args.source.resolve();path=source/'camera.csv';before=hashlib.sha256(path.read_bytes()).hexdigest()
 with path.open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
 arr=lambda k:np.array([float(r[k]) if r.get(k) else np.nan for r in rows])
 t=arr('t_camera');d=arr('index_mcp_object_distance_mm');a=arr('finger_width_mm')
 if not len(d) or not np.isfinite(d[0]) or d[0]<=0 or not np.isfinite(t).all():raise ValueError('First recorded MCP mm distance / timestamps invalid')
 valid=np.array([r.get('valid')=='1' and r.get('index_mcp_distance_valid')=='1' for r in rows])&np.isfinite(d)
 filtered=np.full(len(d),np.nan);state=None
 for i in range(len(d)):
  if not valid[i]:state=None;continue
  if state is None or i==0 or not 0<t[i]-t[i-1]<=.1:state=float(d[i])
  else:
   alpha=1-np.exp(-(t[i]-t[i-1])/args.tau);state+=alpha*(d[i]-state)
  filtered[i]=state
 elapsed=t-t[0];out=BASE/'Report/initial_distance_review'/source.name;out.mkdir(parents=True,exist_ok=True)
 fig,axs=plt.subplots(1,3,figsize=(14,4.3),dpi=180)
 axs[0].plot(elapsed,d,'o-',ms=3,color='#287db5',label='Raw')
 axs[0].plot(elapsed,filtered,'o-',ms=3,color='#e58b26',label='Causal EMA preview')
 axs[0].set_xlim(0,args.initial_seconds)
 selected=(elapsed<=args.initial_seconds)&np.isfinite(d)
 low=min(np.min(d[selected]),np.nanmin(filtered[selected]));high=max(np.max(d[selected]),np.nanmax(filtered[selected]));margin=max(1.,(high-low)*.15)
 axs[0].set_ylim(low-margin,high+margin);axs[0].set(xlabel='Time (s)',ylabel='Index MCP–object distance (mm)',title='Initial distance reversal');axs[0].legend(fontsize=8)
 axs[1].plot(elapsed,a,'o-',ms=3,color='black');axs[1].set_xlim(0,args.initial_seconds)
 selected_a=a[(elapsed<=args.initial_seconds)&np.isfinite(a)];axs[1].set_ylim(max(0,float(selected_a.min())-1),float(selected_a.max())+1)
 axs[1].set(xlabel='Time (s)',ylabel='Finger aperture (mm)',title='Measured aperture: unchanged')
 axs[2].plot(d/d[0],a,color='#287db5',label='Raw')
 axs[2].plot(filtered/d[0],a,color='#e58b26',label='Filtered distance preview')
 axs[2].set(xlabel=r'Normalized distance $l_t/l_0$',ylabel='Finger aperture (mm)',title='Full distance–aperture curves');axs[2].legend(fontsize=8)
 for ax in axs:ax.grid(alpha=.2)
 fig.suptitle(f'{source.name} | raw data preserved; filter preview is not validated correction',fontsize=11);fig.tight_layout()
 for ext in ['png','svg']:fig.savefig(out/f'initial_distance_diagnosis.{ext}',bbox_inches='tight')
 plt.close(fig)
 with (out/'preview_only.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=[*fields,'preview_mcp_distance_mm_ema','preview_mcp_ratio_raw','preview_mcp_ratio_ema']);w.writeheader()
  for i,r in enumerate(rows):w.writerow(dict(r,preview_mcp_distance_mm_ema=filtered[i],preview_mcp_ratio_raw=d[i]/d[0],preview_mcp_ratio_ema=filtered[i]/d[0]))
 first=d[:10];peak=int(np.nanargmax(first))
 report=dict(trial=source.name,raw_first_distance_mm=float(d[0]),first10_max_distance_mm=float(np.nanmax(first)),first10_maximum_time_s=float(elapsed[peak]),initial_outward_excursion_mm=float(np.nanmax(first)-d[0]),initial_raw_max_ratio=float(np.nanmax(first)/d[0]),initial_filtered_max_ratio=float(np.nanmax(filtered[:10])/d[0]),filter='Causal EMA with alpha=1-exp(-dt/tau); reset after invalid input or time gap >0.1s',tau_s=args.tau,normalization='Original first measured distance; no future minimum/maximum or endpoint',raw_csv_unchanged=True,caution='EMA reduces noise but adds response lag; a smooth initial drift may be real movement or landmark drift, not an isolated outlier. Do not enforce monotonic distance without video evidence.')
 assert hashlib.sha256(path.read_bytes()).hexdigest()==before
 (out/'diagnosis.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
