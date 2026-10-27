from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'New/Result/RealTime/Experiment1'

def main():
    data = pd.read_csv(SOURCE/'trial_metrics.csv')
    output = SOURCE/'ppt_large_text'
    output.mkdir(exist_ok=True)
    colors = data.color.map({'red': '#d94c43', 'blue': '#3989c5', 'black': '#555555'}).tolist()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), layout='constrained')
    x = np.arange(len(data))*1.25 + (np.arange(len(data))//3)*.4
    for ax, (column, title, digits) in zip(axes, [('mae', 'MAE (px)', 2), ('rmse', 'RMSE (px)', 2), ('r2', 'R²', 3)]):
        bars = ax.bar(x, data[column], color=colors, width=.78)
        ax.bar_label(bars, labels=[f'{v:.{digits}f}' for v in data[column]], padding=5, fontsize=15)
        ax.set_xticks(x, data.label, rotation=55, ha='right', fontsize=17)
        ax.tick_params(axis='y', labelsize=17)
        ax.set_title(title, fontsize=25, pad=16)
        ax.set_ylim(0, float(data[column].max())*1.17)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    for ext in ['png', 'svg', 'pdf']:
        path = output/f'trial_metrics_large_text.{ext}'
        fig.savefig(path, dpi=300)
        print(path)
    plt.close(fig)

if __name__ == '__main__':
    main()
