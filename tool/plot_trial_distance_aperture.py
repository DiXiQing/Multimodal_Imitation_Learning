"""Export initial-distance-normalized index-MCP distance versus raw aperture."""
from pathlib import Path
import csv
import hashlib
import os
import numpy as np
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).parent / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def export_distance_aperture(rows, destination, unit='mm'):
    column = 'index_mcp_object_distance_mm' if unit == 'mm' else 'index_mcp_object_distance'
    aperture_column = 'finger_width_mm' if unit == 'mm' else 'finger_width'
    distance = np.array([float(r[column]) if r.get(column) else np.nan for r in rows])
    aperture = np.array([float(r[aperture_column]) if r.get(aperture_column) else np.nan for r in rows])
    if not len(distance) or not np.isfinite(distance[0]) or distance[0] <= 0:
        raise ValueError(f'Missing or non-positive initial distance in {column}')
    if not np.any(np.isfinite(distance) & np.isfinite(aperture)):
        raise ValueError('No valid distance/aperture pairs')
    fig, ax = plt.subplots(figsize=(6, 4.3), dpi=200)
    ax.plot(distance / distance[0], aperture, color='#287db5', linewidth=1.5)
    ax.set_xlabel(r'Normalized distance $l_t/l_0$')
    ax.set_ylabel(f'Finger aperture ({unit})')
    ax.grid(alpha=.2)
    fig.tight_layout()
    try:
        for extension in ('png', 'svg'):
            fig.savefig(Path(destination) / f'distance_aperture_normalized.{extension}', bbox_inches='tight')
    finally:
        plt.close(fig)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    args = parser.parse_args()
    folders = sorted(p for p in args.dataset.iterdir()
                     if p.is_dir() and p.name.startswith('grasp_') and p.name[6:].isdigit())
    for folder in folders:
        source = folder / 'camera.csv'
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        with source.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        if not all(r.get('mm_per_px') and float(r['mm_per_px']) > 0 for r in rows):
            raise ValueError(f'Missing millimetre calibration: {folder}')
        export_distance_aperture(rows, folder, 'mm')
        assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    print(f'Exported {len(folders)} normalized distance-aperture plots; CSV files unchanged.')
