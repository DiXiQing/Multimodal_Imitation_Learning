"""Individual raw distance/aperture plots; source files are read only."""
from pathlib import Path
import csv
import hashlib
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1] / 'Data/D2/slow')
parser.add_argument('--distance', choices=['auto', 'unsigned'], default='auto')
args = parser.parse_args()
root = args.source.resolve()
output = root / ('individual_curve_plots_unsigned' if args.distance == 'unsigned' else 'individual_curve_plots')
output.mkdir(exist_ok=True)
count = 0
for folder in sorted(root.iterdir()):
    source = folder / 'camera.csv'
    if not folder.is_dir() or not source.is_file():
        continue
    before = hashlib.sha256(source.read_bytes()).digest()
    with source.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        continue
    def values(key):
        return np.array([float(r[key]) if r.get(key, '').strip() else np.nan for r in rows])
    key = 'hand_object_signed_distance' if args.distance == 'auto' and 'hand_object_signed_distance' in rows[0] else 'hand_object_distance'
    frame = values('frame')
    frame = frame - frame[0] + 1
    distance, aperture = values(key), values('finger_width')
    valid = np.array([r.get('valid', '1') == '1' for r in rows])
    distance[~valid] = np.nan
    aperture[~valid] = np.nan
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(distance, aperture, color='#348ABD', lw=1.7)
    ax.set_xlabel(('Signed hand-object distance' if key == 'hand_object_signed_distance' else 'Hand-object distance') + ' (px)')
    if key == 'hand_object_signed_distance':
        ax.axvline(0, color='gray', ls='--', lw=.7)
    ax.set_ylabel('Fingertip aperture (px)')
    ax.grid(alpha=.2)
    fig.suptitle(folder.name, fontsize=12)
    fig.tight_layout()
    fig.savefig(output / (folder.name + '.png'), dpi=180)
    plt.close(fig)
    assert before == hashlib.sha256(source.read_bytes()).digest()
    count += 1
print(f'Saved {count} individual PNG plots to {output}; original CSV hashes unchanged.')
