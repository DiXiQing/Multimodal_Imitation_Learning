import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

out = Path(__file__).resolve().parent
source = Path(r'C:\Users\Di\.codex\attachments\7e9d182b-162d-412d-96ce-6de41cd51b88\已粘贴的文本.txt')
rows = list(csv.DictReader(source.open(encoding='utf-8-sig')))
x = [float(r['time_s']) for r in rows]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 18, 'axes.labelsize': 22, 'xtick.labelsize': 18, 'ytick.labelsize': 18, 'axes.linewidth': 1.6})
fig, ax = plt.subplots(figsize=(10,10))
for col, label, color in [('actual_aperture_mm','Measured','black'),('function_mm','Function','#3288bd'),('mlp_5f_mm','MLP 5f','#ed9626'),('lstm_full_mm','LSTM Full','#4da157')]:
    ax.plot(x, [float(r[col]) for r in rows], label=label, color=color, linewidth=3.2)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Finger aperture (mm)')
ax.set_xlim(0, max(x))
ax.set_ylim(bottom=0)
ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.grid(True, color='#b0b0b0', alpha=0.18, linewidth=1.5)
ax.set_axisbelow(True)
ax.tick_params(direction='out', length=7, width=1.5, pad=8)
ax.legend(loc='upper right', fontsize=17, frameon=True, facecolor='white', edgecolor='#cccccc')
fig.tight_layout()
for ext in ('png','pdf','svg'):
    fig.savefig(out / ('finger_aperture.' + ext), dpi=300)
print(out / 'finger_aperture.png')
