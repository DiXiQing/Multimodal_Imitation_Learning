"""Plot each cropped aperture curve in a grid without changing measurements."""
import argparse
import csv
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    root = parser.parse_args().root.resolve()
    with (root/'crop_summary.csv').open(encoding='utf-8-sig', newline='') as f:
        summary = list(csv.DictReader(f))
    if not summary:
        raise ValueError('No crop summary entries.')
    columns = 6
    height = math.ceil(len(summary)/columns)
    fig, axes = plt.subplots(height, columns, figsize=(18, 2.45*height), squeeze=False)
    for ax, entry in zip(axes.flat, summary):
        with (root/entry['folder_name']/'camera.csv').open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        ax.plot([float(r['segment_elapsed_s']) for r in rows],
                [float(r['finger_width']) if r['finger_width'] else np.nan for r in rows], lw=1)
        flagged = entry.get('distance_tracking_review') == 'True'
        ax.set_title(entry['folder_name'][-6:]+(' *' if flagged else ''),
                     fontsize=9, color='firebrick' if flagged else 'black')
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(summary):]:
        ax.set_visible(False)
    fig.supxlabel('Segment time (s)')
    fig.supylabel('Fingertip aperture (px)')
    fig.tight_layout(rect=(.02,.02,1,1))
    fig.savefig(root/'aperture_overview.png', dpi=110)
    plt.close(fig)
    print(f'Saved overview: {len(summary)} trials, {root}')

if __name__ == '__main__':
    main()
