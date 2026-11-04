"""Batch process every TRIAL_* folder under Data/BackupData.

For each trial:
  - Read the existing grasp_data.csv.
  - Use its first and last frame as FRAME_START / FRAME_END.
  - Run Analyze_3d_speed.py.
  - Overwrite that trial's grasp_data.csv with the new three-axis speed.

Place this file beside Analyze_3d_speed.py, then run:
    py Batch_Analyze.py
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


BACKUP_DIR = Path(
    r"D:\Code\Multimodal_Imitation_Learning\Data\BackupData"
)

ANALYZE_SCRIPT = Path(__file__).resolve().parent / "Analyze.py"


def read_frame_range(grasp_csv: Path) -> tuple[int, int]:
    frames = []
    with grasp_csv.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or "frame" not in reader.fieldnames:
            raise ValueError("missing frame column")
        for row in reader:
            value = row.get("frame", "")
            if value is None or value.strip() == "":
                continue
            frames.append(int(float(value)))

    if not frames:
        raise ValueError("no valid frame values")
    return min(frames), max(frames)


def main() -> None:
    if not ANALYZE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Analyze_3d_speed.py must be beside this script: {ANALYZE_SCRIPT}"
        )
    if not BACKUP_DIR.exists():
        raise FileNotFoundError(f"BackupData folder not found: {BACKUP_DIR}")

    succeeded = []
    skipped = []
    failed = []

    trial_dirs = sorted(
        path for path in BACKUP_DIR.glob("TRIAL_*") if path.is_dir()
    )

    print(f"Found {len(trial_dirs)} trial folders")
    print("Existing grasp_data.csv files will be overwritten.\n")

    for trial_dir in trial_dirs:
        grasp_csv = trial_dir / "grasp_data.csv"
        camera_csv = trial_dir / "camera.csv"
        imu_csv = trial_dir / "imu.csv"

        missing = [
            path.name
            for path in (grasp_csv, camera_csv, imu_csv)
            if not path.exists()
        ]
        if missing:
            print(f"[Skip] {trial_dir.name}: missing {', '.join(missing)}")
            skipped.append(trial_dir.name)
            continue

        try:
            frame_start, frame_end = read_frame_range(grasp_csv)
        except Exception as error:
            print(f"[Skip] {trial_dir.name}: cannot read frame range: {error}")
            skipped.append(trial_dir.name)
            continue

        print("=" * 70)
        print(
            f"[Analyze] {trial_dir.name}: "
            f"FRAME_START={frame_start}, FRAME_END={frame_end}"
        )

        command = [
            sys.executable,
            str(ANALYZE_SCRIPT),
            "--trial-dir",
            str(trial_dir),
            "--frame-start",
            str(frame_start),
            "--frame-end",
            str(frame_end),
        ]

        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            succeeded.append(trial_dir.name)
        else:
            print(f"[Failed] {trial_dir.name}: exit code {result.returncode}")
            failed.append(trial_dir.name)

    print("\n" + "=" * 70)
    print("Batch analysis finished")
    print(f"Succeeded: {len(succeeded)}")
    print(f"Skipped:   {len(skipped)}")
    print(f"Failed:    {len(failed)}")

    if failed:
        print("Failed trials:")
        for name in failed:
            print(f"  {name}")


if __name__ == "__main__":
    main()
