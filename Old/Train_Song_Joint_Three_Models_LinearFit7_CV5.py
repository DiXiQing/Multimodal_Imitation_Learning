"""Song trial-level, size-stratified five-fold CV using the existing LinearFit7 models."""
import argparse
from datetime import datetime
import random
import numpy as np
import torch
import Train_Song_Joint_Three_Models_LinearFit7_70_30 as base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--seed', type=int, default=20261029)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--check-data', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('epochs must be positive')
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable')
    base.SEED = args.seed
    torch.set_num_threads(min(4, torch.get_num_threads()))
    rng = random.Random(args.seed)
    groups = []
    seen = set()
    for size in base.SIZES:
        folders = sorted(p.parent for p in (base.ROOT / str(size * 10)).glob('202*/camera.csv')
                         if len(p.parent.name) == 15 and p.parent.name.replace('_', '').isdigit())
        trials = [dict(base.load_trial(p), size_cm=size) for p in folders]
        if len(trials) < 5:
            raise ValueError(f'{size * 10} mm: fewer than five trials')
        for trial in trials:
            digest = trial['csv_sha256']
            if digest in seen:
                raise ValueError(f"Duplicate CSV: {trial['folder']}")
            seen.add(digest)
        rng.shuffle(trials)
        groups.append([list(chunk) for chunk in np.array_split(np.array(trials, dtype=object), 5)])
        print(f'{size * 10} mm: {len(trials)} trials', flush=True)
    folds = []
    for k in range(5):
        test = [g for group in groups for g in group[k]]
        train = [g for group in groups for j, chunk in enumerate(group) if j != k for g in chunk]
        folds.append((train, test))
        print(f'Fold {k+1}: {len(train)} train / {len(test)} test', flush=True)
    if args.check_data:
        for train, test in folds:
            base.configure_velocity(train, test, args)
            base.check_data(test)
        print('Five-fold data checks passed; no training performed.', flush=True)
        return
    out = base.ROOT / 'Result' / f'Joint_Three_Models_LinearFit7_SongJoint_CV5_{datetime.now():%Y%m%d_%H%M%S}'
    out.mkdir(parents=True, exist_ok=False)
    base.write_json(out/'split_manifest.json', dict(source=str(base.ROOT), seed=args.seed,
        epochs=args.epochs, evaluation='size-stratified trial-level 5-fold CV',
        velocity_method='LinearFit7', max_future_frames=0, spline_lambda=base.SPLINE_LAMBDA,
        summary_unit='out-of-fold trial; mean and SD across trials',
        folds=[{label:[dict(trial=g['name'], size_cm=g['size_cm'], csv_sha256=g['csv_sha256']) for g in seq]
                for label, seq in [('Train',train),('Test',test)]} for train,test in folds]))
    metrics = []
    for number, (train, test) in enumerate(folds, 1):
        print(f'Training fold {number}/5', flush=True)
        random.seed(args.seed + number)
        np.random.seed(args.seed + number)
        torch.manual_seed(args.seed + number)
        fold_out = out / f'fold_{number:02d}'
        fold_out.mkdir()
        base.configure_velocity(train, test, args)
        args.lstm_train, static_records = base.extend_lstm_training(train, args.velocity_scaling)
        base.write_json(fold_out/'velocity_scaling.json', args.velocity_scaling)
        base.write_csv(fold_out/'lstm_static_prefixes.csv', static_records)
        current = base.run_holdout(train, test, args, fold_out)
        for row in current:
            row['fold'] = number
        metrics.extend(current)
        summary = base.summarize(current)
        base.write_csv(fold_out/'test_mean_sd.csv', summary)
        base.plot_summary(fold_out, summary)
        base.plot_training(fold_out)
        base.plot_trials(fold_out)
    base.write_csv(out/'test_per_trial_metrics.csv', metrics)
    summary = base.summarize(metrics)
    base.write_csv(out/'test_mean_sd.csv', summary)
    base.plot_summary(out, summary)
    print(f'Saved {out}', flush=True)


if __name__ == '__main__':
    main()
