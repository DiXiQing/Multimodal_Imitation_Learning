"""Compare Euclidean and signed-distance MLPs with identical causal context.

Uses the existing training implementation. Running this file trains two new
models; --check-data validates aligned inputs without training or writing results.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import train_distance_window_mlp as core

SOURCE = r'C:\MineApp\Code\Multimodal_Imitation_Learning\Data\D2'
WINDOW_SIZE = 10  # Both models use this context; set 1 for no history.
SPECS = {'distance_unsigned': 'hand_object_distance',
         'distance_signed': 'hand_object_signed_distance'}
LABELS = {'distance_unsigned': 'Old distance', 'distance_signed': 'Signed distance'}
COLORS = ['#3485b8', '#e59a32']
PHASES = [('rest', 'Initial'), ('movement', 'Grasping'), ('hold', 'Hold')]


def aligned_partitions(source, gap):
    datasets = {name: core.load_folder_partitions(source, gap, column)
                for name, column in SPECS.items()}
    # Use only targets valid for BOTH complete distance windows, independently
    # within each trial. Do this before fitting scalers and computing weights.
    for part in (0, 1):
        maps = {name: {g['name']: g for g in data[part]} for name, data in datasets.items()}
        names = list(maps['distance_unsigned'])
        if set(names) != set(maps['distance_signed']):
            raise ValueError('Distance definitions have different trial sets.')
        for name in names:
            old, new = [maps[model][name] for model in SPECS]
            common = np.intersect1d(old['frame'], new['frame'])
            if not len(common):
                raise ValueError(f'{name}: no shared valid target frames.')
            for trial in (old, new):
                keep = np.isin(trial['frame'], common)
                for key in ('windows', 'y', 'frame', 'time', 'window_start_frame', 'window_duration', 'phase'):
                    trial[key] = trial[key][keep]
            for key in ('frame', 'y', 'time', 'phase', 'window_start_frame'):
                if not np.array_equal(old[key], new[key]):
                    raise ValueError(f'{name}: target alignment failed: {key}')
    return datasets


def save_figure(fig, path):
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(path.with_suffix('.' + ext), dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--window-size', type=int, default=WINDOW_SIZE)
    p.add_argument('--epochs', type=int, default=core.EPOCHS)
    p.add_argument('--patience', type=int, default=core.PATIENCE)
    p.add_argument('--batch-size', type=int, default=core.BATCH_SIZE)
    p.add_argument('--lr', type=float, default=core.LEARNING_RATE)
    p.add_argument('--weight-decay', type=float, default=core.WEIGHT_DECAY)
    p.add_argument('--hidden', type=int, default=core.HIDDEN_SIZE)
    p.add_argument('--loss', choices=['mae', 'mse'], default=core.LOSS)
    p.add_argument('--seed', type=int, default=core.SEED)
    p.add_argument('--max-gap', type=float, default=core.MAX_FRAME_GAP_S)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--log-every', type=int, default=25)
    p.add_argument('--output', type=Path)
    p.add_argument('--check-data', action='store_true')
    args = p.parse_args()
    if min(args.window_size, args.epochs, args.patience, args.batch_size, args.hidden, args.log_every) < 1 or args.lr <= 0 or args.max_gap <= 0 or args.weight_decay < 0:
        p.error('Invalid window size or training parameters.')
    core.configure_window(args.window_size)
    source = args.source.resolve()
    datasets = aligned_partitions(source, args.max_gap)
    fit, validation = datasets['distance_unsigned']
    print(f'Aligned inputs: {len(fit)} Train / {len(validation)} Validation; context={args.window_size}', flush=True)
    if args.check_data:
        for name, (train, val) in datasets.items():
            x = np.concatenate([g['windows'].ravel() for g in train + val])
            print(f'{name}: {SPECS[name]}, range {x.min():.3f}..{x.max():.3f} px')
        print('Data check only. No training or output files.')
        return
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable.')
    device = torch.device('cuda' if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    out = args.output or source/'Result'/f'distance_definition_window{args.window_size}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    out.mkdir(parents=True, exist_ok=False)
    (out/'models').mkdir()
    (out/'validation_predictions').mkdir()
    def log(message=''):
        print(message, flush=True)
        with (out/'training_log.txt').open('a', encoding='utf-8') as f:
            f.write(message+'\n')
    info = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    info.update(distance_columns=SPECS, input_units=args.window_size, output_units=1,
                input_order='Oldest to current signed/Euclidean distance; no aperture history.',
                output='Current-frame fingertip aperture in px.',
                hidden_layers=[args.hidden, args.hidden],
                comparison='Same trial split, common valid target frames, seed, architecture and hyperparameters. Each scaler uses aligned Train only.',
                evaluation='Validation selects checkpoints; not independent testing.')
    (out/'model_information.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    (out/'split.json').write_text(json.dumps(dict(train=[g['name'] for g in fit], validation=[g['name'] for g in validation]), indent=2), encoding='utf-8')
    core.write_csv(out/'data_split.csv', [dict(trial=g['name'], split=part, raw_rows=g['raw_rows'], eligible_target_frames=len(g['y']))
                   for part, data in [('train', fit), ('validation', validation)] for g in data])
    trained, histories, selections = {}, [], []
    for name, column in SPECS.items():
        args.distance_column = column
        train, val = datasets[name]
        log(f'{LABELS[name]}: {column}; {args.window_size} input units -> 1 output unit')
        model, scaler, history, selection = core.train_one(name, args.window_size, train, val, args, out, device, log)
        trained[name] = (model, scaler)
        histories.extend(history)
        selections.append(selection)
    core.write_csv(out/'training_history.csv', histories)
    core.write_csv(out/'epoch_selection.csv', selections)
    predictions, metrics = [], []
    for j, g in enumerate(validation):
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(g['time'], g['y'], color='#222222', lw=2, label='Measured')
        pred = {}
        for (name, (model, scaler)), color in zip(trained.items(), COLORS):
            trial = datasets[name][1][j]
            pred[name] = core.predict_trial(model, trial, args.window_size, scaler, device)
            metrics.append(dict(model=name, trial=g['name'], frames=len(g['y']), **core.scores(g['y'], pred[name])))
            ax.plot(g['time'], pred[name], color=color, label=LABELS[name])
        trial_rows = []
        for i in range(len(g['y'])):
            row = dict(trial=g['name'], source_frame=int(g['frame'][i]), time_s=float(g['time'][i]), phase=str(g['phase'][i]),
                       actual_aperture_px=float(g['y'][i]), old_distance_px=float(g['windows'][i,-1]),
                       signed_distance_px=float(datasets['distance_signed'][1][j]['windows'][i,-1]),
                       predicted_unsigned_px=float(pred['distance_unsigned'][i]), predicted_signed_px=float(pred['distance_signed'][i]))
            trial_rows.append(row)
        predictions.extend(trial_rows)
        core.write_csv(out/'validation_predictions'/f'{g["name"]}.csv', trial_rows)
        ax.set(title=g['name'], xlabel='Time from segment start (s)', ylabel='Aperture (px)')
        ax.legend(); ax.grid(alpha=.2)
        save_figure(fig, out/'validation_predictions'/g['name'])
    core.write_csv(out/'validation_predictions.csv', predictions)
    core.write_csv(out/'validation_trial_metrics.csv', metrics)
    summary = []
    phase_trials, phase_summary = [], []
    for name, key in [('distance_unsigned','predicted_unsigned_px'), ('distance_signed','predicted_signed_px')]:
        y = np.array([r['actual_aperture_px'] for r in predictions])
        yp = np.array([r[key] for r in predictions])
        summary.append(dict(model=name, frames=len(y), **core.scores(y, yp)))
        for phase, _ in PHASES:
            for g in validation:
                rows = [r for r in predictions if r['trial']==g['name'] and r['phase']==phase]
                if rows:
                    phase_trials.append(dict(model=name, trial=g['name'], phase=phase, frames=len(rows),
                        **core.scores(np.array([r['actual_aperture_px'] for r in rows]), np.array([r[key] for r in rows]))))
            part = [r for r in phase_trials if r['model']==name and r['phase']==phase]
            vals = [r['mae'] for r in part]
            phase_summary.append(dict(model=name, phase=phase, trials=len(vals), mean_trial_mae_px=float(np.mean(vals)) if vals else None,
                                      sd_trial_mae_px=float(np.std(vals,ddof=1)) if len(vals)>1 else 0.))
    core.write_csv(out/'overall_model_summary.csv', summary)
    core.write_csv(out/'validation_trial_phase_metrics.csv', phase_trials)
    core.write_csv(out/'validation_phase_summary.csv', phase_summary)
    # Explicit distance labels replace the old Current/Window comparison labels.
    fig, axes = plt.subplots(1,3,figsize=(12,4.5))
    for ax,key,title in zip(axes,['mae','rmse','r2'],['MAE (px)','RMSE (px)','R²']):
        bars=ax.bar(list(LABELS.values()), [r[key] if r[key] is not None else np.nan for r in summary], color=COLORS)
        ax.bar_label(bars,fmt='%.3f' if key=='r2' else '%.2f',padding=3)
        ax.set_title(title);ax.margins(y=.2);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    save_figure(fig,out/'validation_model_comparison')
    fig,axes=plt.subplots(1,3,figsize=(14,max(4.5,.45*(len(validation)+1)+2)))
    pos=np.arange(len(validation)+1)
    for ax,key,title in zip(axes,['mae','rmse','r2'],['MAE (px)','RMSE (px)','R²']):
        for j,(name,color) in enumerate(zip(SPECS,COLORS)):
            by_trial={r['trial']:r for r in metrics if r['model']==name}
            overall=next(r for r in summary if r['model']==name)
            values=[by_trial[g['name']][key] for g in validation]+[overall[key]]
            ax.barh(pos+(j-.5)*.34,[v if v is not None else np.nan for v in values],height=.32,color=color,label=LABELS[name])
        ax.set_yticks(pos,[f'Validation {i:02d}' for i in range(1,len(validation)+1)]+['Overall'])
        ax.invert_yaxis();ax.set_title(title);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(fontsize=8)
    save_figure(fig,out/'validation_metrics_chart')
    core.write_csv(out/'validation_label_mapping.csv',[dict(label=f'Validation {i:02d}',trial=g['name']) for i,g in enumerate(validation,1)])
    fig,ax=plt.subplots(figsize=(8,5.5))
    for j,(name,color) in enumerate(zip(SPECS,COLORS)):
        rows=[r for r in phase_summary if r['model']==name]
        ax.bar(np.arange(3)+(j-.5)*.34,[r['mean_trial_mae_px'] if r['mean_trial_mae_px'] is not None else np.nan for r in rows],
               width=.32,yerr=[r['sd_trial_mae_px'] for r in rows],capsize=5,color=color,label=LABELS[name])
    ax.set_xticks(range(3),[label for _,label in PHASES]);ax.set_ylabel('Mean per-grasp MAE (px)');ax.legend();ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    save_figure(fig,out/'validation_phase_errors')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,name in zip(axes,SPECS):
        rows=[r for r in histories if r['model']==name]
        for key,label in [('train_mae_px','Train (dropout active)'),('validation_mae_px','Validation')]:
            ax.plot([r['epoch'] for r in rows],[r[key] for r in rows],label=label)
        ax.set(title=LABELS[name],xlabel='Epoch',ylabel='MAE (px)');ax.legend();ax.grid(alpha=.2)
    save_figure(fig,out/'training_curves')
    for r in summary:log(str(r))
    log(f'Saved to {out}')
    (out/'RESULT.txt').write_text((out/'training_log.txt').read_text(encoding='utf-8'),encoding='utf-8')


if __name__ == '__main__':
    main()
