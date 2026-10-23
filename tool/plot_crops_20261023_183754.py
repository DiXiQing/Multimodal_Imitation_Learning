import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] / 'Data/TopView/TRIAL_Color_black_Size_3cm_20261023_183754_203502/grasp_segments'
with (ROOT/'crop_summary.csv').open(encoding='utf-8-sig', newline='') as f:
    summary = list(csv.DictReader(f))
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
grid, cells = plt.subplots(9, 6, figsize=(19, 23))
manifest = []
for idx, entry in enumerate(summary):
    folder = ROOT / entry['folder_name']
    with (folder/'camera.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    group = 'Fast' if idx < 28 else 'Slow'
    color = '#348ABD' if group == 'Fast' else '#E59A2B'
    t = np.array([float(r['segment_elapsed_s']) for r in rows])
    a = np.array([float(r['finger_width']) if r['finger_width'] else np.nan for r in rows])
    d = np.array([float(r['hand_object_distance']) if r['hand_object_distance'] else np.nan for r in rows])
    s = np.array([float(r['hand_object_signed_distance']) if r['hand_object_signed_distance'] else np.nan for r in rows])
    label = group if idx in (0, 28) else None
    axes[0].plot(t, a, color=color, alpha=.45, lw=1, label=label)
    axes[1].plot(t, s, color=color, alpha=.45, lw=1)
    axes[2].plot(s, a, color=color, alpha=.45, lw=1)
    ax = cells.flat[idx]
    ax.plot(t, a, color=color, lw=1.2)
    ax.set_title(f'{idx+1:02d} {group} | {folder.name[-6:]}', fontsize=9)
    ax.set_ylim(0, 125)
    ax.grid(alpha=.18)
    ax.tick_params(labelsize=7)
    meta = json.loads((folder/'segment.json').read_text())
    onset, stable = int(meta['onset'])-int(meta['start']), int(meta['stable'])-int(meta['start'])
    manifest.append(dict(folder_name=folder.name, group=group, frames=len(rows), duration_s=t[-1], onset_to_stable_s=t[stable]-t[onset], minimum_signed_distance_px=float(np.nanmin(s))))
for ax in list(cells.flat)[len(summary):]:
    ax.set_visible(False)
for ax, x, y in zip(axes, ['Segment time (s)', 'Segment time (s)', 'Signed hand-object distance (px)'], ['Fingertip aperture (px)', 'Signed hand-object distance (px)', 'Fingertip aperture (px)']):
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(alpha=.18)
axes[0].legend(frameon=False)
axes[1].axhline(0, color='gray', lw=.7)
axes[2].axvline(0, color='gray', lw=.7)
fig.tight_layout()
fig.savefig(ROOT/'fast_slow_overlay.png', dpi=200)
fig.savefig(ROOT/'fast_slow_overlay.pdf')
grid.supxlabel('Segment time (s)')
grid.supylabel('Fingertip aperture (px)')
grid.tight_layout(rect=(.015,.015,1,1))
grid.savefig(ROOT/'aperture_overview.png', dpi=140)
with (ROOT/'speed_groups.csv').open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(manifest[0]))
    writer.writeheader()
    writer.writerows(manifest)
for group in ('Fast','Slow'):
    values = [r['onset_to_stable_s'] for r in manifest if r['group']==group]
    print(group, len(values), 'onset-to-stable mean/SD', np.mean(values), np.std(values, ddof=1))
print('Plots and speed_groups.csv saved:', ROOT)
