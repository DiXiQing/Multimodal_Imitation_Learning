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
    parser=argparse.ArgumentParser(description='Overlay normalized distance versus aperture for three sizes; raw CSVs read-only.')
    parser.add_argument('--source',type=Path,default=PROJECT_ROOT/'Data/1101_D');parser.add_argument('--output',type=Path)
    args=parser.parse_args();root=args.source.resolve();out=(args.output or root/'distance_aperture_combined').resolve();out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(7,5.7),dpi=240);counts={};hashes={};records=[]
    for size,color in [(20,'#d62728'),(30,'#287db5'),(40,'#222222')]:
        folders=sorted(p for p in (root/f'{size}mm').iterdir() if p.is_dir() and re.fullmatch(r'(?:\d{8}_\d{6}|grasp_\d+)',p.name));counts[str(size)]=len(folders)
        for folder in folders:
            path=folder/'camera.csv';hashes[path]=hashlib.sha256(path.read_bytes()).hexdigest()
            with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
            settings=json.loads((folder/'settings.json').read_text(encoding='utf-8-sig'));scale=(settings.get('pixel_calibration') or {}).get('mm_per_px')
            if scale is None or not np.isfinite(float(scale)) or float(scale)<=0:raise ValueError(f'{folder}: missing calibration')
            d=np.array([float(r['index_mcp_object_distance_mm'] or 'nan') for r in rows]);a=np.array([float(r['finger_width_mm'] or 'nan') for r in rows])
            if not np.isfinite(d[0]) or d[0]<=0:raise ValueError(f'{folder}: invalid initial distance')
            ax.plot(d/d[0],a,color=color,alpha=.18,lw=.65)
            records.append(dict(size_mm=size,trial=folder.name,frames=len(rows),initial_distance_mm=float(d[0]),csv_sha256=hashes[path]))
        ax.plot([],[],color=color,lw=1.7,label=f'{size} mm (n={len(folders)})')
        print(f'{size} mm: {len(folders)} trials',flush=True)
    ax.set_xlabel(r'Normalized distance $l_t/l_0$',fontsize=12);ax.set_ylabel(r'$d$ (mm)',fontsize=12);ax.grid(alpha=.17);ax.set_axisbelow(True);ax.legend(fontsize=10);fig.tight_layout()
    for ext in ('png','svg','pdf'):fig.savefig(out/f'three_sizes_distance_aperture.{ext}',bbox_inches='tight')
    plt.close(fig)
    for path,before in hashes.items():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=before:raise RuntimeError(f'CSV changed: {path}')
    with (out/'included_trials.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (out/'plot_information.json').write_text(json.dumps(dict(counts=counts,total_trials=sum(counts.values()),x='index_mcp_object_distance_mm / first recorded value',y='finger_width_mm',colors={'20mm':'red','30mm':'blue','40mm':'black'},smoothing=False,raw_data_modified=False),indent=2),encoding='utf-8');print('Saved:',out)


if __name__=='__main__':main()
