"""Generate 24 synthetic reach-to-grasp trials from the real D2 dataset.

The generated files keep the same grasp_data.csv schema as the real data:
frame, t_camera, finger_width, object_area, ax, ay, az, mag, velocity
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SOURCE_ROOT = Path(r"D:\Code\Multimodal_Imitation_Learning\Data\临时数据\D2")
OUTPUT_ROOT = Path(r"D:\Code\Multimodal_Imitation_Learning\Data\临时数据\D3")
COLORS = ("Black", "Blue", "Red")
TRIALS_PER_COLOR = 8
FRAME_COUNT = 60
RESERVED_FRAMES = 15
FPS = 30.0
SEED = 20261006


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def smooth_noise(rng: np.random.Generator, length: int, scale: float) -> np.ndarray:
    raw = rng.normal(0.0, scale, length)
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
    kernel /= kernel.sum()
    return np.convolve(raw, kernel, mode="same")


def load_reference_stats() -> dict[str, dict[str, float]]:
    stats = {}
    for color in COLORS:
        rows = []
        for csv_path in sorted(SOURCE_ROOT.glob(f"TRIAL_{color}_*/grasp_data.csv")):
            frame = pd.read_csv(csv_path)
            required = ["finger_width", "object_area", "velocity"]
            if any(column not in frame.columns for column in required):
                continue
            frame[required] = frame[required].apply(pd.to_numeric, errors="coerce")
            frame = frame.dropna(subset=required).reset_index(drop=True)
            if len(frame) < 10:
                continue
            rows.append(
                {
                    "area_start": float(frame["object_area"].iloc[:5].median()),
                    "area_final": float(frame["object_area"].iloc[-5:].median()),
                    "width_start": float(frame["finger_width"].iloc[:5].median()),
                    "width_max": float(frame["finger_width"].max()),
                    "width_final": float(frame["finger_width"].iloc[-5:].median()),
                    "velocity_peak": float(frame["velocity"].max()),
                }
            )
        if len(rows) < 2:
            raise RuntimeError(f"Not enough reference trials for {color}")
        stats[color] = pd.DataFrame(rows).median(numeric_only=True).to_dict()
    return stats


def make_trial(color: str, trial_index: int, reference: dict[str, float], rng: np.random.Generator) -> pd.DataFrame:
    n = FRAME_COUNT
    frame = np.arange(1000 + trial_index * 100, 1000 + trial_index * 100 + n, dtype=int)
    time = 2000000000.0 + trial_index * 10.0 + np.arange(n, dtype=float) / FPS

    # Small trial-to-trial scale changes, kept within the natural range seen in D2.
    area_start = reference["area_start"] * rng.uniform(0.95, 1.05)
    area_final = reference["area_final"] * rng.uniform(0.93, 1.07)
    width_start = max(8.0, reference["width_start"] + rng.normal(0.0, 3.0))
    width_max = reference["width_max"] * rng.uniform(0.96, 1.04)
    width_final = reference["width_final"] * rng.uniform(0.94, 1.06)
    velocity_peak = reference["velocity_peak"] * rng.uniform(0.92, 1.08)

    # Movement begins after the 15-frame MLP context window.
    movement = np.clip((np.arange(n) - RESERVED_FRAMES) / (n - RESERVED_FRAMES - 1), 0.0, 1.0)
    progress = smoothstep(movement)

    # Object area grows monotonically and smoothly toward the final visual size.
    area = area_start + (area_final - area_start) * progress
    area += smooth_noise(rng, n, max(4.0, area_final * 0.0025))
    area[:RESERVED_FRAMES] = area_start + smooth_noise(rng, RESERVED_FRAMES, max(2.0, area_start * 0.0015))
    area = np.maximum(area, 1.0)
    area[RESERVED_FRAMES:] = np.maximum.accumulate(area[RESERVED_FRAMES:])

    # Aperture peaks near the middle of reaching, then closes smoothly.
    peak_u = float(np.clip(rng.normal(0.48, 0.025), 0.40, 0.56))
    close_u = float(np.clip(rng.normal(0.78, 0.025), 0.68, 0.88))
    opening = smoothstep(movement / peak_u)
    closing = smoothstep((movement - peak_u) / max(close_u - peak_u, 0.1))
    aperture = width_start + (width_max - width_start) * opening
    aperture -= (width_max - width_final) * closing
    aperture += smooth_noise(rng, n, 1.6)
    aperture[:RESERVED_FRAMES] = width_start + smooth_noise(rng, RESERVED_FRAMES, 0.8)
    aperture = np.clip(aperture, 3.0, None)

    # Single-peaked hand speed; its peak is close to the aperture peak.
    speed_center = float(np.clip(peak_u + rng.uniform(-0.055, 0.055), 0.38, 0.62))
    speed_sigma = float(rng.uniform(0.15, 0.19))
    velocity = velocity_peak * np.exp(-0.5 * ((movement - speed_center) / speed_sigma) ** 2)
    velocity += smooth_noise(rng, n, max(0.004, velocity_peak * 0.015))
    velocity[:RESERVED_FRAMES] = np.abs(smooth_noise(rng, RESERVED_FRAMES, 0.003))
    velocity = np.clip(velocity, 0.0, None)
    velocity[-3:] *= np.linspace(0.65, 0.05, 3)

    # Build a physically coherent 3-D velocity vector and differentiate it.
    direction_phase = rng.uniform(-0.15, 0.15)
    direction = np.column_stack(
        [
            np.cos(direction_phase + 0.16 * movement),
            0.35 * np.sin(direction_phase + 0.16 * movement),
            0.15 * np.cos(direction_phase + 0.11 * movement),
        ]
    )
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    velocity_vector = velocity[:, None] * direction
    acceleration = np.gradient(velocity_vector, 1.0 / FPS, axis=0)
    acceleration += smooth_noise(rng, n, 0.025)[:, None] * rng.normal(0.0, 1.0, (1, 3))
    acceleration[:RESERVED_FRAMES] *= 0.15

    result = pd.DataFrame(
        {
            "frame": frame,
            "t_camera": time,
            "finger_width": aperture,
            "object_area": area,
            "ax": acceleration[:, 0],
            "ay": acceleration[:, 1],
            "az": acceleration[:, 2],
            "mag": np.linalg.norm(acceleration, axis=1),
            "velocity": velocity,
        }
    )
    return result


def main() -> None:
    if not SOURCE_ROOT.is_dir():
        raise SystemExit(f"Reference dataset not found: {SOURCE_ROOT}")

    reference_stats = load_reference_stats()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    created = []
    trial_index = 0
    for color in COLORS:
        for number in range(1, TRIALS_PER_COLOR + 1):
            trial_index += 1
            trial_name = f"TRIAL_{color}_SIM_{number:02d}"
            trial_dir = OUTPUT_ROOT / trial_name
            trial_dir.mkdir(parents=True, exist_ok=True)
            frame = make_trial(color, trial_index, reference_stats[color], rng)
            frame.to_csv(trial_dir / "grasp_data.csv", index=False, float_format="%.6f")
            created.append(trial_name)

    print(f"Created {len(created)} trials in {OUTPUT_ROOT}")
    for color in COLORS:
        print(f"  {color}: {TRIALS_PER_COLOR}")
    print(f"Reference stats: {reference_stats}")


if __name__ == "__main__":
    main()
