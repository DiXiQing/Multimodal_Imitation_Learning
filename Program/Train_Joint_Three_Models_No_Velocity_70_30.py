"""Quick joint three-model comparison without velocity, using one 70/30 split."""

import argparse
from datetime import datetime
import random

import torch

import Train_Joint_Three_Models_BackwardDiff_Mean4_CV5 as joint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--check-data', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable')
    args.without_velocity = True
    torch.set_num_threads(min(4, torch.get_num_threads()))

    reference_train, reference_test = joint.partition()[0]
    trials = reference_train + reference_test
    rng = random.Random(joint.SEED)
    train, test = [], []
    for size in joint.SIZES:
        group = sorted((trial for trial in trials if trial['size_cm'] == size),
                       key=lambda trial: trial['name'])
        rng.shuffle(group)
        cutoff = round(.7 * len(group))
        train.extend(group[:cutoff])
        test.extend(group[cutoff:])
        print(f'{size * 10} mm: {cutoff} train / {len(group) - cutoff} test', flush=True)
    if args.check_data:
        return

    joint.configure_velocity(train, test, args)
    output = joint.ROOT / 'Result' / f'Joint_Three_Models_No_Velocity_70_30_{datetime.now():%Y%m%d_%H%M%S}'
    output.mkdir(exist_ok=False)
    joint.write_json(output / 'split_manifest.json', {
        'seed': joint.SEED,
        'epochs': args.epochs,
        'features': ['l/l0', 'size_cm'],
        'common_support': 'Fifth through last frame of each contiguous valid run',
        **{label: [{'trial': trial['name'], 'size_cm': trial['size_cm'],
                   'csv_sha256': trial['csv_sha256']} for trial in group]
           for label, group in [('Train', train), ('Test', test)]},
    })
    evaluation = output / 'holdout'
    metrics = joint.run_fold(train, test, args, evaluation, 1)
    summary = joint.summarize(metrics)
    joint.write_csv(output / 'test_mean_sd.csv', summary)
    joint.plot_summary(output, summary)
    joint.plot_training(evaluation)
    joint.plot_trials(evaluation)
    print(f'Saved {output}', flush=True)


if __name__ == '__main__':
    main()
