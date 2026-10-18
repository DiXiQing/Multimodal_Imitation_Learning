from pathlib import Path
from datetime import datetime
import argparse
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_window_mlp as base

DELTA_WEIGHT = 1.0


def paired_windows(trials, names, features, window_size):
    parts = []
    for name in names:
        x, y, frames = base.make_trial_windows(trials[name], features, window_size)
        previous_x = np.concatenate([x[:1], x[:-1]])
        previous_y = np.concatenate([y[:1], y[:-1]])
        valid = np.concatenate([[False], np.diff(frames) == 1])
        parts.append((x, y, previous_x, previous_y, valid))
    if not parts:
        raise ValueError('No training trials')
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(5))


def delta_loss(model, x, previous_x, y, previous_y, valid):
    if not valid.any():
        return x.new_zeros(())
    dropouts = [module for module in model.modules() if isinstance(module, nn.Dropout)]
    states = [module.training for module in dropouts]
    try:
        for module in dropouts:
            module.train(False)
        current, previous = model(torch.cat([x[valid], previous_x[valid]])).chunk(2)
        return nn.functional.mse_loss(current - previous, y[valid] - previous_y[valid])
    finally:
        for module, state in zip(dropouts, states):
            module.train(state)


def train(model_name, features, trials, train_names, args, device, weight, output):
    base.set_seed(args.seed)
    x, y, px, py, valid = paired_windows(trials, train_names, features, args.window_size)
    xm, xs, ym, ys = base.normalization_from_training(x, y)
    tensors = [torch.from_numpy(v.astype(np.float32)) for v in [(x-xm)/xs, (y-ym)/ys, (px-xm)/xs, (py-ym)/ys]]
    loader = DataLoader(TensorDataset(*tensors, torch.from_numpy(valid)), batch_size=args.batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    model = base.WindowMLP(args.window_size, len(features), args.hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = []
    print(f'{output.name}/{model_name}: {len(x)} windows, {int(valid.sum())} adjacent pairs, lambda={weight:g}', flush=True)
    for epoch in range(1, args.epochs+1):
        model.train()
        total_point = total_delta = 0.
        pair_count = 0
        for batch in loader:
            bx, by, bpx, bpy, mask = [v.to(device) for v in batch]
            optimizer.zero_grad()
            point = nn.functional.mse_loss(model(bx), by)
            change = delta_loss(model, bx, bpx, by, bpy, mask) if weight else bx.new_zeros(())
            loss = point + weight*change
            loss.backward()
            optimizer.step()
            total_point += point.item()*len(bx)
            total_delta += change.item()*int(mask.sum())
            pair_count += int(mask.sum())
        point_mean = total_point/len(x)
        delta_mean = total_delta/max(1, pair_count)
        history.append(dict(epoch=epoch, point_mse=point_mean, delta_mse=delta_mean,
                            total_loss=point_mean+weight*delta_mean, delta_weight=weight))
        if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
            print(f'{output.name}/{model_name} epoch {epoch}: MSE={point_mean:.5f}, delta={delta_mean:.5f}', flush=True)
    directory = output/'models'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/f'{model_name}.pth'
    torch.save(dict(model_name=model_name, model_class='WindowMLP', state_dict=model.cpu().state_dict(),
        feature_columns=features, target_column=base.TARGET_COLUMN, window_size=args.window_size,
        feature_count=len(features), hidden_size=args.hidden_size, x_mean=torch.tensor(xm),
        x_scale=torch.tensor(xs), y_mean=float(ym), y_scale=float(ys), train_trials=train_names,
        train_windows=len(x), train_adjacent_pairs=int(valid.sum()), epochs=args.epochs,
        delta_weight=weight, objective='MSE + lambda * adjacent target-difference MSE',
        delta_units='target-standardized FW change per consecutive recorded frame',
        delta_dropout='disabled only for deterministic difference branch', seed=args.seed), path)
    pd.DataFrame(history).to_csv(output/f'{model_name}_training_history.csv', index=False)
    return path


def evaluate(trials, names, paths, args, device, folder, predictor=None):
    predictor = predictor or base.predict_trial
    output = args.output_dir/folder
    output.mkdir(exist_ok=True)
    records = []
    for name in names:
        frame = trials[name]
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.plot(frame.frame, frame.finger_width, color='black', lw=2.5, label='Measured FW')
        predictions = None
        for label, path in paths.items():
            target, prediction, frames = predictor(path, frame, device)
            rmse, mae, r2 = base.regression_metrics(target, prediction)
            adjacent = np.diff(frames) == 1
            slope_error = np.diff(prediction)[adjacent] - np.diff(target)[adjacent]
            target_peak = int(np.argmax(target)); predicted_peak = int(np.argmax(prediction))
            records.append(dict(trial=name, object_size_cm=base.trial_size(name), model=label,
                rmse_px=rmse, mae_px=mae, r2=r2,
                delta_rmse_px_per_frame=float(np.sqrt(np.mean(slope_error**2))) if len(slope_error) else np.nan,
                peak_amplitude_error_px=float(prediction[predicted_peak]-target[target_peak]),
                peak_frame_error=float(frames[predicted_peak]-frames[target_peak])))
            ax.plot(frames, prediction, lw=1.5, label=f'{label}: RMSE {rmse:.1f} px')
            if predictions is None:
                predictions = pd.DataFrame(dict(frame=frames, measured_fw=target))
            assert np.array_equal(predictions.frame, frames)
            predictions[label] = prediction
        predictions.to_csv(output/f'{name}.csv', index=False)
        ax.set(title=name, xlabel='Original frame', ylabel='FW (px)');ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(output/f'{name}.png', dpi=150);plt.close(fig)
    metrics = pd.DataFrame(records)
    metrics.to_csv(args.output_dir/f'{folder}_metrics.csv', index=False)
    summary = metrics.groupby('model')[['rmse_px','mae_px','r2','delta_rmse_px_per_frame']].agg(['mean','std'])
    summary.to_csv(args.output_dir/f'{folder}_summary.csv')
    print(folder, '\n', summary.to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description='WindowMLP: compare MSE with MSE + adjacent FW difference loss')
    parser.add_argument('--mode', choices=['single_size','generalization'], default=base.RUN_MODE)
    parser.add_argument('--data-root', type=Path, default=base.DATA_ROOT)
    parser.add_argument('--generalization-root', type=Path, default=base.GENERALIZATION_ROOT)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--split-csv', type=Path, help='Reuse a saved split exactly; fail if a referenced trial is missing')
    parser.add_argument('--delta-weight', type=float, default=DELTA_WEIGHT)
    for name, value, kind in [('window-size',base.WINDOW_SIZE,int),('epochs',base.EPOCHS,int),
        ('batch-size',base.BATCH_SIZE,int),('lr',base.LEARNING_RATE,float),
        ('weight-decay',base.WEIGHT_DECAY,float),('hidden-size',base.HIDDEN_SIZE,int),
        ('seed',base.SEED,int),('test-trials',base.TEST_TRIALS,int)]:
        parser.add_argument('--'+name, type=kind, default=value)
    parser.add_argument('--test-names', nargs='+', default=base.TEST_TRIAL_NAMES)
    args = parser.parse_args()
    if not np.isfinite(args.delta_weight) or args.delta_weight < 0:parser.error('delta-weight must be finite and nonnegative')
    if min(args.epochs,args.batch_size,args.hidden_size,args.test_trials)<1 or args.window_size<2:parser.error('Invalid training dimensions')
    if args.lr<=0 or args.weight_decay<0:parser.error('Invalid optimizer settings')
    try:
        if args.split_csv:
            if args.test_names:raise ValueError('Use either split-csv or test-names')
            split = pd.read_csv(args.split_csv)
            if split.trial.duplicated().any():raise ValueError('Duplicate trial in split')
            if not set(split.split)<= {'train','test','generalization_test'}:raise ValueError('Unknown split labels')
            if args.mode=='single_size':
                trials=base.load_trials(args.data_root)
                if len({base.trial_size(n) for n in trials})!=1:raise ValueError('single_size requires one size')
            else:
                trials={}
                if set(base.TRAIN_SIZE_FOLDERS)&set(base.GENERALIZATION_SIZE_FOLDERS):raise ValueError('Overlapping sizes')
                for folder in (*base.TRAIN_SIZE_FOLDERS,*base.GENERALIZATION_SIZE_FOLDERS):
                    part=base.load_trials(args.generalization_root/folder)
                    if set(trials)&set(part):raise ValueError('Duplicate trials across sizes')
                    if any(base.trial_size(n)!=float(folder) for n in part):raise ValueError('Size/folder mismatch')
                    trials.update(part)
                allowed={float(s) for s in base.TRAIN_SIZE_FOLDERS}
                if any(base.trial_size(r.trial) not in allowed for r in split.itertuples() if r.split in {'train','test'}):raise ValueError('Generalization size used for training/test')
            missing=set(split.trial)-set(trials)
            if missing:raise ValueError(f'Split references unavailable trials: {sorted(missing)}')
            train_names=split.loc[split.split=='train','trial'].tolist()
            test_names=split.loc[split.split=='test','trial'].tolist()
            generalization_names=split.loc[split.split=='generalization_test','trial'].tolist()
        else:
            trials,train_names,test_names,generalization_names=base.prepare_split(args)
        if not train_names or not test_names:raise ValueError('Require training and test trials')
    except (ValueError,RuntimeError,KeyError,OSError) as exc:parser.error(str(exc))
    directory=args.generalization_root if args.mode=='generalization' else args.data_root
    args.output_dir=args.output_dir or directory/'Result'/f'temporalLoss_{datetime.now():%Y%m%d_%H%M%S_%f}'
    if args.output_dir.exists():parser.error('Output directory already exists; choose a new directory')
    args.output_dir.mkdir(parents=True)
    base.save_split(train_names,test_names,args.output_dir,generalization_names)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    paths={}
    for objective,weight in [('MSE',0.),('MSE_delta',args.delta_weight)]:
        output=args.output_dir/objective;output.mkdir()
        for model_name,features in base.MODEL_SPECS.items():
            paths[f'{objective} | {base.MODEL_LABELS[model_name]}']=train(model_name,features,trials,train_names,args,device,weight,output)
    evaluate(trials,test_names,paths,args,device,'test_predictions')
    if generalization_names:evaluate(trials,generalization_names,paths,args,device,'generalization_predictions')
    print('Saved:',args.output_dir,flush=True)


if __name__=='__main__':main()
