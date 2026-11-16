"""Train an LSTM with synthetic stationary frames before each training trial."""

import argparse
from datetime import datetime
import random

import numpy as np
import torch

import Train_Joint_Three_Models_BackwardDiff_Mean4_CV5 as joint


def prepend_static(trial, seconds):
    if seconds == 0:
        return trial, 0
    first = trial['seqs'][0]
    if first[0] != joint.WINDOW - 1:
        return trial, 0

    times = trial['t']
    dt = float(np.median(np.diff(times[:joint.WINDOW])))
    count = round(seconds / dt)
    if count == 0:
        return trial, 0
    distance = float(np.median(trial['d_mm'][:joint.WINDOW]))
    aperture = float(np.median(trial['y'][:joint.WINDOW]))
    padded_distance = np.r_[np.full(count, distance), trial['d_mm']]
    padded_times = np.r_[times[0] - dt*np.arange(count, 0, -1), times]
    runs = [np.arange(first[-1] + count + 1)]
    runs.extend(np.arange(run[0] - joint.WINDOW + 1, run[-1] + 1) + count
                for run in trial['seqs'][1:])
    targets = [run[joint.WINDOW - 1:] for run in runs]
    augmented = dict(
        trial, d=padded_distance / distance, d_mm=padded_distance,
        initial_distance_mm=distance, t=padded_times,
        y=np.r_[np.full(count, aperture), trial['y']],
        seqs=targets, common=np.concatenate(targets),
        velocity_mm_s=joint.estimate_velocity(padded_distance, padded_times, runs),
    )
    return augmented, count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--static-seconds', nargs='+', type=float, default=[0, .5, 1, 2, 3])
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--check-data', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or any(not np.isfinite(s) or s < 0 for s in args.static_seconds):
        parser.error('Use positive epochs and finite, nonnegative static durations.')
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(joint.SEED)

    reference_train, reference_test = joint.partition()[0]
    trials = reference_train + reference_test
    split_rng = random.Random(joint.SEED)
    original_train, original_test = [], []
    for size in joint.SIZES:
        group = sorted((trial for trial in trials if trial['size_cm'] == size),
                       key=lambda trial: trial['name'])
        split_rng.shuffle(group)
        cutoff = round(.7 * len(group))
        original_train.extend(group[:cutoff])
        original_test.extend(group[cutoff:])
        print(f'{size * 10} mm: {cutoff} train / {len(group)-cutoff} test', flush=True)
    rng = random.Random(joint.SEED + 1)
    train, manifest = [], []
    for original in original_train:
        seconds = rng.choice(args.static_seconds)
        trial, frames = prepend_static(original, seconds)
        train.append(dict(trial))
        manifest.append(dict(trial=trial['name'], size_cm=trial['size_cm'],
                             csv_sha256=trial['csv_sha256'],
                             requested_static_seconds=seconds, added_frames=frames))
    test = [dict(trial) for trial in original_test]
    args.without_velocity = False
    joint.configure_velocity(train, test, args)
    for trial, original, record in zip(train, original_train, manifest):
        frames = record['added_frames']
        np.testing.assert_array_equal(trial['y'][frames:], original['y'])
        if frames >= joint.WINDOW:
            np.testing.assert_array_equal(
                trial['velocity_mm_s'][joint.WINDOW - 1:frames], 0)
    padded = sum(row['added_frames'] > 0 for row in manifest)
    print(f'70/30: {len(train)} training trials ({padded} padded), '
          f'{len(test)} original test trials', flush=True)
    if args.check_data:
        return

    output = joint.ROOT / 'Result' / f'LSTM_BackwardDiff_Mean4_Static_70_30_{datetime.now():%Y%m%d_%H%M%S}'
    output.mkdir(exist_ok=False)
    joint.write_json(output / 'split_manifest.json', dict(
        train_fraction=.7, epochs=args.epochs, seed=joint.SEED,
        static_seed=joint.SEED + 1,
        static_seconds=args.static_seconds, Train=manifest,
        Test=[dict(trial=g['name'], size_cm=g['size_cm'], csv_sha256=g['csv_sha256']) for g in test],
    ))
    joint.write_json(output / 'velocity_scaling.json', args.velocity_scaling)
    targets = np.concatenate([g['y'][g['common']] for g in train])
    mean, std = float(targets.mean()), max(float(targets.std()), 1e-6)
    model = joint.JointLSTM(True).to(args.device)
    history = joint.fit(model, 'LSTM Full', train, mean, std, args.epochs, args.device)
    joint.write_csv(output / 'training_history.csv', history)
    joint.plot_training(output)
    torch.save(dict(state_dict=model.state_dict(), mean_mm=mean, std_mm=std,
                    epochs=args.epochs, velocity_method=joint.VELOCITY_METHOD,
                    velocity_scaling=args.velocity_scaling), output / 'lstm_full.pth')

    metrics = []
    with torch.inference_mode():
        for trial in test:
            predicted = np.full(len(trial['t']), np.nan)
            for indices, inputs in joint.batches(trial, 'LSTM Full', args.device):
                predicted[indices] = model(inputs).reshape(-1).cpu().numpy()*std + mean
            common = trial['common']
            metrics.append(dict(trial=trial['name'], size_cm=trial['size_cm'], model='LSTM Full',
                                **joint.scores(trial['y'][common], predicted[common])))
            path = output / 'Test' / f"{trial['size_cm']}cm" / f"{trial['name']}.csv"
            joint.write_csv(path, [
                dict(frame=frame, time_s=time, actual_mm=actual, lstm_full_mm=estimate)
                for frame, time, actual, estimate in zip(trial['frame'], trial['t'], trial['y'], predicted)
            ])
    joint.write_csv(output / 'test_per_trial_metrics.csv', metrics)
    joint.write_csv(output / 'test_mean_sd.csv',
                    [row for row in joint.summarize(metrics) if row['model'] == 'LSTM Full'])
    print(f'Saved {output}', flush=True)


if __name__ == '__main__':
    main()
