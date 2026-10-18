from pathlib import Path
from datetime import datetime
import argparse
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_window_mlp as base
from train_window_mlp_temporal import evaluate

REFERENCE_SPLIT = ROOT/'Data/Exprement Data/Khairy/3/Result/temporalLoss_20261016_140009_768410/data_split.csv'


class CausalConv1d(nn.Module):
    def __init__(self, inputs, outputs, dilation):
        super().__init__()
        self.left_padding = 2*dilation
        self.conv = nn.Conv1d(inputs, outputs, kernel_size=3, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.left_padding, 0)))


class CausalWindowCNN(nn.Module):
    def __init__(self, window_size, feature_count, channels=16, hidden_size=32, dropout=.1):
        super().__init__()
        layers = []
        receptive_field, dilation, inputs = 1, 1, feature_count
        while receptive_field < window_size:
            layers.extend([CausalConv1d(inputs, channels, dilation), nn.ReLU(), nn.Dropout(dropout)])
            receptive_field += 2*dilation
            dilation *= 2
            inputs = channels
        self.receptive_field = receptive_field
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(channels, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))

    def forward(self, x):
        encoded = self.encoder(x.transpose(1, 2))
        return self.head(encoded[:, :, -1]).squeeze(-1)


def build_model(architecture, window_size, features, hidden_size, channels, dropout):
    if architecture == 'MLP':
        return base.WindowMLP(window_size, features, hidden_size)
    return CausalWindowCNN(window_size, features, channels, hidden_size, dropout)


def train(architecture, name, features, trials, train_names, args, device):
    base.set_seed(args.seed)
    x, y = base.make_windows(trials, train_names, features, args.window_size)
    xm, xs, ym, ys = base.normalization_from_training(x, y)
    dataset = TensorDataset(torch.from_numpy((x-xm)/xs), torch.from_numpy((y-ym)/ys))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    model = build_model(architecture, args.window_size, len(features), args.hidden_size, args.channels, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    parameters = sum(p.numel() for p in model.parameters())
    receptive = model.receptive_field if architecture == 'CNN' else args.window_size
    print(f'{architecture}/{name}: {len(x)} windows, {parameters} parameters, receptive field={receptive}, MSE only', flush=True)
    history = []
    for epoch in range(1, args.epochs+1):
        model.train()
        total = 0.
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(bx), by)
            loss.backward()
            optimizer.step()
            total += loss.item()*len(bx)
        history.append(dict(epoch=epoch, mse=total/len(x)))
        if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
            print(f'{architecture}/{name} epoch {epoch}/{args.epochs}: MSE={total/len(x):.5f}', flush=True)
    directory = args.output_dir/architecture
    (directory/'models').mkdir(parents=True, exist_ok=True)
    path = directory/'models'/f'{name}.pth'
    torch.save(dict(model_class='CausalWindowCNN' if architecture == 'CNN' else 'WindowMLP',
        architecture=architecture, model_name=name, state_dict=model.cpu().state_dict(),
        feature_columns=features, target_column=base.TARGET_COLUMN, window_size=args.window_size,
        feature_count=len(features), hidden_size=args.hidden_size, channels=args.channels,
        dropout=args.dropout if architecture=='CNN' else .1, kernel_size=3, receptive_field=receptive,
        x_mean=torch.tensor(xm), x_scale=torch.tensor(xs), y_mean=float(ym), y_scale=float(ys),
        train_trials=train_names, train_windows=len(x), epochs=args.epochs, seed=args.seed,
        objective='MSE', parameter_count=parameters), path)
    pd.DataFrame(history).to_csv(directory/f'{name}_training_history.csv', index=False)
    return path


def predict_trial(path, frame, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    x, target, frames = base.make_trial_windows(frame, checkpoint['feature_columns'], checkpoint['window_size'])
    x = (x-checkpoint['x_mean'].cpu().numpy())/checkpoint['x_scale'].cpu().numpy()
    model = build_model(checkpoint['architecture'], checkpoint['window_size'], checkpoint['feature_count'],
                        checkpoint['hidden_size'], checkpoint['channels'], checkpoint['dropout']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.inference_mode():
        prediction = model(torch.from_numpy(x).to(device)).cpu().numpy()*checkpoint['y_scale']+checkpoint['y_mean']
    return target.astype(float), prediction.astype(float), frames


def load_split(args):
    if args.split_csv is None:
        return base.prepare_split(args)
    if args.test_names:
        raise ValueError('Use either split-csv or test-names')
    split = pd.read_csv(args.split_csv)
    if split.trial.duplicated().any() or not set(split.split) <= {'train','test','generalization_test'}:
        raise ValueError('Invalid split rows')
    if args.mode == 'single_size':
        trials = base.load_trials(args.data_root)
        if len({base.trial_size(n) for n in trials}) != 1:
            raise ValueError('single_size requires one size')
        if (split.split == 'generalization_test').any():
            raise ValueError('Single-size mode cannot reuse a generalization split')
    else:
        if set(base.TRAIN_SIZE_FOLDERS) & set(base.GENERALIZATION_SIZE_FOLDERS):
            raise ValueError('Training and generalization sizes overlap')
        trials = {}
        for folder in (*base.TRAIN_SIZE_FOLDERS, *base.GENERALIZATION_SIZE_FOLDERS):
            part = base.load_trials(args.generalization_root/folder)
            if set(trials) & set(part) or any(base.trial_size(n) != float(folder) for n in part):
                raise ValueError('Duplicate trials or mismatched size folders')
            trials.update(part)
        for row in split.itertuples():
            allowed = base.GENERALIZATION_SIZE_FOLDERS if row.split == 'generalization_test' else base.TRAIN_SIZE_FOLDERS
            if base.trial_size(row.trial) not in {float(s) for s in allowed}:
                raise ValueError('Split does not match training/generalization sizes')
    missing = set(split.trial)-set(trials)
    if missing:
        raise ValueError(f'Split references missing trials; no silent substitution: {sorted(missing)}')
    names = [split.loc[split.split == key, 'trial'].tolist() for key in ['train','test','generalization_test']]
    return trials, *names


def main():
    parser = argparse.ArgumentParser(description='Small causal 1D CNN versus WindowMLP, both trained with MSE')
    parser.add_argument('--mode', choices=['single_size','generalization'], default=base.RUN_MODE)
    parser.add_argument('--data-root', type=Path, default=base.DATA_ROOT)
    parser.add_argument('--generalization-root', type=Path, default=base.GENERALIZATION_ROOT)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--split-csv', type=Path)
    parser.add_argument('--new-split', action='store_true', help='Generate a new split rather than reuse the preceding experiment')
    parser.add_argument('--channels', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=.1)
    for name,value,kind in [('window-size',base.WINDOW_SIZE,int),('epochs',base.EPOCHS,int),
        ('batch-size',base.BATCH_SIZE,int),('lr',base.LEARNING_RATE,float),('weight-decay',base.WEIGHT_DECAY,float),
        ('hidden-size',base.HIDDEN_SIZE,int),('seed',base.SEED,int),('test-trials',base.TEST_TRIALS,int)]:
        parser.add_argument('--'+name, type=kind, default=value)
    parser.add_argument('--test-names', nargs='+', default=None)
    args = parser.parse_args()
    if args.new_split and args.split_csv:parser.error('new-split conflicts with split-csv')
    if not args.new_split and args.split_csv is None and args.mode=='single_size' and args.data_root.resolve()==REFERENCE_SPLIT.parents[2].resolve() and not args.test_names:
        args.split_csv=REFERENCE_SPLIT
    if min(args.epochs,args.batch_size,args.hidden_size,args.channels,args.test_trials)<1 or args.window_size<2:
        parser.error('Invalid training dimensions')
    if not np.isfinite([args.lr,args.weight_decay,args.dropout]).all() or args.lr<=0 or args.weight_decay<0 or not 0<=args.dropout<1:
        parser.error('Invalid optimizer/dropout settings')
    try:
        trials, train_names, test_names, generalization_names = load_split(args)
        if not train_names or not test_names:raise ValueError('Require training and test trials')
    except (ValueError,RuntimeError,KeyError,OSError) as exc:parser.error(str(exc))
    directory=args.generalization_root if args.mode=='generalization' else args.data_root
    args.output_dir=args.output_dir or directory/'Result'/f'causalCNN_{datetime.now():%Y%m%d_%H%M%S_%f}'
    if args.output_dir.exists():parser.error('Choose a new output directory')
    args.output_dir.mkdir(parents=True)
    base.save_split(train_names,test_names,args.output_dir,generalization_names)
    print('Split:', args.split_csv or 'newly generated', flush=True)
    print(f'Train={len(train_names)}, test={len(test_names)}, generalization={len(generalization_names)}', flush=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    paths={}
    for architecture in ['MLP','CNN']:
        for name,features in base.MODEL_SPECS.items():
            paths[f'{architecture} | {base.MODEL_LABELS[name]}']=train(architecture,name,features,trials,train_names,args,device)
    evaluate(trials,test_names,paths,args,device,'test_predictions',predictor=predict_trial)
    if generalization_names:evaluate(trials,generalization_names,paths,args,device,'generalization_predictions',predictor=predict_trial)
    print('Saved:',args.output_dir,flush=True)


if __name__=='__main__':main()
