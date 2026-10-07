"""Generate synthetic trials for object sizes not seen during training.

Training reference sizes:
    Red   = 2.0 cm
    Blue  = 3.0 cm
    Black = 4.5 cm

Generated unseen sizes:
    2.5 cm, 3.5 cm, 4.0 cm

The output keeps the same grasp_data.csv schema as D3 and is written to a
separate directory so it cannot accidentally enter the D3 training set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from generate_synthetic_grasp_data import (
    COLORS,
    FRAME_COUNT,
    FPS,
    RESERVED_FRAMES,
    make_trial,
    load_reference_stats,
)


OUTPUT_ROOT = Path(
    r"D:\Code\Multimodal_Imitation_Learning\Data\临时数据\D4_UnseenSizes"
)
TRIALS_PER_SIZE = 8
SEED = 20261007

SIZE_BY_COLOR = {
    "Red": 2.0,
    "Blue": 3.0,
    "Black": 4.5,
}

UNSEEN_SIZES = (2.5, 3.5, 4.0)


def interpolate_reference(
    reference_stats: dict[str, dict[str, float]],
    target_size: float,
) -> dict[str, float]:
    anchors = sorted(
        (SIZE_BY_COLOR[color], reference_stats[color])
        for color in COLORS
    )

    if not anchors[0][0] <= target_size <= anchors[-1][0]:
        raise ValueError(f"Target size outside reference range: {target_size}")

    for (left_size, left_stats), (right_size, right_stats) in zip(
        anchors[:-1], anchors[1:]
    ):
        if left_size <= target_size <= right_size:
            ratio = (target_size - left_size) / (right_size - left_size)
            return {
                key: float(left_stats[key] + ratio * (right_stats[key] - left_stats[key]))
                for key in left_stats
            }

    raise RuntimeError(f"Could not interpolate size {target_size}")


def size_label(size_cm: float) -> str:
    return f"{size_cm:g}".replace(".", "p")


def main() -> None:
    reference_stats = load_reference_stats()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    metadata = []
    trial_index = 100

    for size_cm in UNSEEN_SIZES:
        reference = interpolate_reference(reference_stats, size_cm)
        label = size_label(size_cm)

        for number in range(1, TRIALS_PER_SIZE + 1):
            trial_index += 1
            trial_name = f"TRIAL_Size_{label}cm_SIM_{number:02d}"
            trial_dir = OUTPUT_ROOT / trial_name
            trial_dir.mkdir(parents=True, exist_ok=True)

            frame = make_trial("UnseenSize", trial_index, reference, rng)
            frame.to_csv(
                trial_dir / "grasp_data.csv",
                index=False,
                float_format="%.6f",
            )

            metadata.append(
                {
                    "trial": trial_name,
                    "object_size_cm": size_cm,
                    "split": "unseen_test",
                    "frame_count": len(frame),
                    "window_reserved_frames": RESERVED_FRAMES,
                    "fps": FPS,
                    "source_reference": "D2 Red/Blue/Black interpolated",
                }
            )

    pd.DataFrame(metadata).to_csv(
        OUTPUT_ROOT / "trial_metadata.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Created {len(metadata)} trials in {OUTPUT_ROOT}")
    for size_cm in UNSEEN_SIZES:
        print(f"  {size_cm:g} cm: {TRIALS_PER_SIZE}")


if __name__ == "__main__":
    main()
