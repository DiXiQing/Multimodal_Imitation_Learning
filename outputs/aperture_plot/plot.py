from pathlib import Path
import csv
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

base = Path(__file__).resolve().parent
with (base / 'data.csv').open(encoding='utf-8-sig') as f:
    rows = list(csv.DictReader(f))
t = [float(r['time_s']) for r in rows]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 18,
                     'axes.labelsize': 22, 'xtick.labelsize': 18,
                     'ytick.labelsize': 18, 'axes.linewidth': 1.5})
fig, ax = plt.subplots(figsize=(10, 10))
for col, label, color in [
    ('actual_aperture_mm', 'Measured', 'black'),
    ('function_mm', 'Function', '#348abd'),
    ('mlp_5f_mm', 'MLP 5f', '#e89627'),
    ('lstm_full_mm', 'LSTM Full', '#509f56'),
]:
    ax.plot(t, [float(r[col]) for r in rows], label=label, color=color, linewidth=3)
ax.set(xlabel='Time (s)', ylabel='Finger aperture (mm)', xlim=(0, max(t)))
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.set_axisbelow(True)
ax.grid(True, color='#dedede', alpha=0.5, linewidth=1.4)
ax.tick_params(direction='out', length=7, width=1.5, pad=8)
ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=18)
fig.tight_layout()
fig.savefig(base / 'finger_aperture.png', dpi=300)
fig.savefig(base / 'finger_aperture.svg')
print(f'{len(rows)} rows; time 0–{max(t):.3f} s')
