"""Scatter plot of trial duration vs initial MCP distance; raw data read-only."""
from pathlib import Path
import argparse,csv,json,re,hashlib,os
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).parent/'.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
PROJECT_ROOT=next((p for p in Path(__file__).resolve().parents
                   if (p/'Data').is_dir() and (p/'New/Program').is_dir()),Path(__file__).resolve().parent)
DATA_ROOT = r'C:\MineApp\Code\Multimodal_Imitation_Learning\Data\1101_D\30mm'


def load_conditions(source):
    folders=sorted(p for p in source.iterdir() if p.is_dir() and re.fullmatch(r'(?:\d{8}_\d{6}|grasp_\d+)',p.name))
    if not folders:raise ValueError('No active experiment folders found')
    result=[];hashes={}
    for folder in folders:
        path=folder/'camera.csv'
        hashes[path]=hashlib.sha256(path.read_bytes()).hexdigest()
        meta=json.loads((folder/'segment.json').read_text(encoding='utf-8-sig'))
        settings=json.loads((folder/'settings.json').read_text(encoding='utf-8-sig'))
        calibration=(settings.get('pixel_calibration') or {}).get('mm_per_px')
        if calibration is None or not np.isfinite(float(calibration)) or float(calibration)<=0:
            raise ValueError(f'{folder.name}: missing millimetre calibration; px fallback forbidden')
        with path.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f);fields=reader.fieldnames or [];rows=list(reader)
        if not rows or any(k not in fields for k in ('frame','t_camera','index_mcp_object_distance_mm')):
            raise ValueError(f'{folder.name}: empty CSV or missing frame/timestamp/MCP distance column')
        if any(meta.get(k) is None for k in ('start_row','onset_row','closure_confirmed_row')):
            raise ValueError(f'{folder.name}: missing confirmed action boundary; duration cannot be inferred safely')
        onset=int(meta['onset_row'])-int(meta['start_row'])
        closure=int(meta['closure_confirmed_row'])-int(meta['start_row'])
        if not 0<=onset<closure<len(rows):raise ValueError(f'{folder.name}: boundaries outside current retained CSV')
        t=np.array([float(r['t_camera']) for r in rows])
        if not np.isfinite(t).all() or np.any(np.diff(t)<=0):raise ValueError(f'{folder.name}: invalid/non-increasing timestamps')
        initial=float(rows[0]['index_mcp_object_distance_mm'])
        if not np.isfinite(initial) or initial<=0:raise ValueError(f'{folder.name}: invalid initial MCP distance')
        if rows[0].get('valid','1')!='1' or rows[0].get('index_mcp_distance_valid','1')!='1':
            raise ValueError(f'{folder.name}: invalid initial detection')
        if 'index_mcp_object_distance' in fields:
            expected=float(rows[0]['index_mcp_object_distance'])*float(calibration)
            if not np.isclose(initial,expected,rtol=1e-5,atol=1e-4):raise ValueError(f'{folder.name}: distance inconsistent with calibration')
        result.append(dict(trial=folder.name,grasp_duration_s=float(t[closure]-t[onset]),initial_distance_mm=initial,clip_duration_s=float(t[-1]-t[0]),action_onset_source_frame=int(rows[onset]['frame']),closure_confirmed_source_frame=int(rows[closure]['frame']),source=str(path)))
    return result,hashes


def save_plot(rows,output,source):
    if (output/'camera.csv').exists():raise ValueError('Output must be a separate analysis folder, not an experiment folder')
    output.mkdir(parents=True,exist_ok=True)
    x=np.array([r['grasp_duration_s'] for r in rows]);y=np.array([r['initial_distance_mm'] for r in rows])
    fig,ax=plt.subplots(figsize=(7.2,5.3),dpi=240)
    fig.subplots_adjust(left=.14,right=.97,bottom=.15,top=.91)
    ax.scatter(x,y,s=40,color='#287db5',alpha=.75,edgecolors='white',linewidths=.5,zorder=3)
    for v in set([float(x.min()),float(x.max())]):ax.axvline(v,color='#7B8590',ls='--',lw=.85)
    for v in set([float(y.min()),float(y.max())]):ax.axhline(v,color='#7B8590',ls='--',lw=.85)
    ax.set_xlabel('trial duration(s)',fontsize=12);ax.set_ylabel('Initial hand–object distance (mm)',fontsize=12)
    ax.grid(alpha=.16);ax.set_axisbelow(True)
    dx=max(float(np.ptp(x)),.1);dy=max(float(np.ptp(y)),2.)
    ax.set_xlim(max(0,float(x.min())-.12*dx),float(x.max())+.12*dx)
    ax.set_ylim(float(y.min())-.13*dy,float(y.max())+.30*dy)
    fig.tight_layout()
    for ext in ('png','svg','pdf'):fig.savefig(output/f'duration_initial_distance_scatter.{ext}',bbox_inches='tight')
    plt.close(fig)
    with (output/'per_trial_conditions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    report=report_info(rows,source)
    (output/'definitions_and_ranges.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def report_info(rows,source):
    return dict(source=str(source),trials=len(rows),duration_definition='Actual t_camera at annotated five-frame closure confirmation minus t_camera at annotated action onset; pre/post margins excluded',distance_definition='First recorded index_mcp_object_distance_mm; Euclidean landmark5 to object centre; no time/distance normalization',duration_range_s=[min(r['grasp_duration_s'] for r in rows),max(r['grasp_duration_s'] for r in rows)],initial_distance_range_mm=[min(r['initial_distance_mm'] for r in rows),max(r['initial_distance_mm'] for r in rows)],excluded='backup, review_segments, Train, Test, Result and derived folders; discovery is not recursive',raw_data_modified=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path(DATA_ROOT))
    parser.add_argument('--output',type=Path,help='Default: source/condition_coverage_current')
    parser.add_argument('--check-data',action='store_true',help='Read-only validation, no plots or files created')
    args=parser.parse_args();source=args.source.resolve()
    if not source.is_dir():parser.error(f'Dataset not found: {source}')
    try:
        rows,hashes=load_conditions(source)
        report=report_info(rows,source)
        if not args.check_data:
            output=(args.output or source/'condition_coverage_current').resolve()
            save_plot(rows,output,source);print(f'Saved scatter plot and per-trial values: {output}')
        for p,before in hashes.items():
            if hashlib.sha256(p.read_bytes()).hexdigest()!=before:raise RuntimeError(f'CSV changed during analysis: {p}')
    except (OSError,ValueError,KeyError,TypeError) as exc:parser.error(str(exc))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
