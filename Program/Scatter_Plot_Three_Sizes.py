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


def main():
    parser=argparse.ArgumentParser(description='Combined condition coverage: 20mm red, 30mm blue, 40mm black. Raw data read-only.')
    parser.add_argument('--source',type=Path,default=PROJECT_ROOT/'Data/1101_D')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();root=args.source.resolve()
    groups=[];hashes={};rows=[]
    for size,color in [(20,'#d62728'),(30,'#287db5'),(40,'#222222')]:
        group,checks=load_conditions(root/f'{size}mm');hashes.update(checks)
        groups.append((size,color,group));rows.extend(dict(size_mm=size,**r) for r in group)
        print(f'{size} mm: {len(group)} trials',flush=True)
    output=(args.output or root/'condition_coverage_combined').resolve();output.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(8,5.7),dpi=240)
    for size,color,group in groups:
        ax.scatter([r['grasp_duration_s'] for r in group],[r['initial_distance_mm'] for r in group],s=32,color=color,alpha=.65,edgecolors='white',linewidths=.35,label=f'{size} mm (n={len(group)})')
    ax.set_xlabel('trial duration(s)',fontsize=12);ax.set_ylabel('Initial hand–object distance (mm)',fontsize=12)
    ax.grid(alpha=.17);ax.set_axisbelow(True);ax.legend(fontsize=10);fig.tight_layout()
    for ext in ('png','svg','pdf'):fig.savefig(output/f'three_sizes_condition_scatter.{ext}',bbox_inches='tight')
    plt.close(fig)
    with (output/'per_trial_conditions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for path,before in hashes.items():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=before:raise RuntimeError(f'CSV changed: {path}')
    (output/'plot_information.json').write_text(json.dumps(dict(total_trials=len(rows),counts={f'{size}mm':len(group) for size,_,group in groups},colors={'20mm':'red','30mm':'blue','40mm':'black'},duration_definition='Actual closure-confirmation timestamp minus annotated action-onset timestamp; margins excluded',distance_definition='First recorded index_mcp_object_distance_mm; no normalization',excluded='backup, Train, Test, Result, review_segments and derived folders; immediate trial folders only',raw_data_modified=False),indent=2),encoding='utf-8')
    print('Saved:',output,flush=True)


if __name__=='__main__':main()
