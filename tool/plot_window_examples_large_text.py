from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'New/Data/D1/2/Result/distance_mlp_window5_20261018_230453_976437/validation_predictions'
OUTPUT = ROOT/'Zemi/assets/window_examples_large_text'
TRIALS = ['Red_2cm_20261017_202604', 'Red_2cm_20261017_202608', 'Red_2cm_20261017_202611']

def draw(ax, data):
    for column, label, color in [('actual_aperture_px', 'Measured', '#222222'), ('predicted_current_px', 'Current distance', '#2686ca'), ('predicted_window5_px', '5-frame window', '#f28e16')]:
        ax.plot(data.time_s, data[column], label=label, color=color, lw=2.8)
    ax.axvspan(.81, .95, color='#df6666', alpha=.17, zorder=0)
    ax.set_ylim(0, 100)
    ax.set_xlim(.08, 1.7)
    ax.set_xlabel('Time from segment start (s)', fontsize=21, labelpad=10)
    ax.set_ylabel('Aperture (px)', fontsize=21, labelpad=10)
    ax.tick_params(labelsize=19)
    ax.xaxis.set_major_locator(MultipleLocator(.4))
    ax.grid(alpha=.18)

def save(fig, name):
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(OUTPUT/f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})
    data = [pd.read_csv(SOURCE/f'{name}.csv') for name in TRIALS]
    for name, frame in zip(TRIALS, data):
        fig, ax = plt.subplots(figsize=(7.5, 8))
        draw(ax, frame)
        ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02), ncol=3, frameon=False, fontsize=13, handlelength=1.3, columnspacing=.8, handletextpad=.3)
        fig.tight_layout()
        save(fig, name)
    fig, axes = plt.subplots(1, 3, figsize=(21, 8))
    for ax, frame in zip(axes, data):
        draw(ax, frame)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=3, frameon=False, fontsize=23, bbox_to_anchor=(.5, 1))
    fig.tight_layout(rect=(0, 0, 1, .92))
    save(fig, 'three_window_examples_large_text')
    pd.DataFrame({'trial': TRIALS, 'source_csv': [str(SOURCE/f'{n}.csv') for n in TRIALS], 'highlight_start_s': .81, 'highlight_end_s': .95}).to_csv(OUTPUT/'sources.csv', index=False)
    print(OUTPUT)

if __name__ == '__main__':
    main()
