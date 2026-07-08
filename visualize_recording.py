"""
visualize_recording.py
多模态录制数据的加载与可视化（Camera + EMG + Servo angle）

Usage:
    python visualize_recording.py
    python visualize_recording.py --recording recording_20260701_212500
    python visualize_recording.py --recording recording_20260701_212500 --summary
    python visualize_recording.py --recording recording_20260701_212500 --save overview.png
    python visualize_recording.py --list

Also importable:
    from visualize_recording import load_recording
    rec = load_recording("recording_20260701_212500")
    emg = rec.load_emg(0)   # (100, 4) float32
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

# Match Record2.py
EMG_FS = 1000
EMG_WINDOW_MS = 100
EMG_CHANNELS = 4
EMG_CHANNEL_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

BASE_DIR = Path(__file__).resolve().parent
META_COLS = ["timestamp", "frame_index", "angle", "angle_timestamp", "frame_file", "emg_file"]


@dataclass
class Recording:
    """One recording session loaded from a recording_* folder."""

    path: Path
    timestamp: np.ndarray
    frame_index: np.ndarray
    angle: np.ndarray              # float, NaN where missing
    angle_timestamp: np.ndarray    # float, NaN where missing
    frame_file: list[str]
    emg_file: list[str]
    emg_rms: np.ndarray            # (N, 4) per-frame RMS
    duration_s: float
    fps: float

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def n_frames(self) -> int:
        return len(self.frame_index)

    def frame_path(self, idx: int) -> Path | None:
        fname = self.frame_file[idx]
        if not fname:
            return None
        p = self.path / "frames" / fname
        return p if p.exists() else None

    def emg_path(self, idx: int) -> Path | None:
        fname = self.emg_file[idx]
        if not fname:
            return None
        p = self.path / "emg" / fname
        return p if p.exists() else None

    def load_emg(self, idx: int) -> np.ndarray | None:
        p = self.emg_path(idx)
        if p is None:
            return None
        return np.load(p)

    def get_angle(self, idx: int) -> float | None:
        val = self.angle[idx]
        return None if np.isnan(val) else float(val)


def find_recordings(root: Path | None = None) -> list[Path]:
    root = root or BASE_DIR
    return sorted(root.glob("recording_*"))


def find_csv(rec_dir: Path) -> Path:
    csvs = sorted(rec_dir.glob("data_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No data_*.csv in {rec_dir}")
    return csvs[0]


def _parse_float(val: str) -> float:
    val = (val or "").strip()
    if not val:
        return np.nan
    try:
        return float(val)
    except ValueError:
        return np.nan


def _load_metadata(csv_path: Path) -> dict:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({k: row.get(k, "") for k in META_COLS if k in row})
        if not rows and reader.fieldnames:
            # fallback: read whatever meta cols exist
            f.seek(0)
            reader = csv.DictReader(f)
            rows = list(reader)

    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")

    n = len(rows)
    timestamp = np.array([_parse_float(r.get("timestamp", "")) for r in rows], dtype=np.float64)
    frame_index = np.array([int(float(r.get("frame_index", i))) for i, r in enumerate(rows)], dtype=np.int64)
    angle = np.array([_parse_float(r.get("angle", "")) for r in rows], dtype=np.float64)
    angle_timestamp = np.array([_parse_float(r.get("angle_timestamp", "")) for r in rows], dtype=np.float64)
    frame_file = [r.get("frame_file", "") or "" for r in rows]
    emg_file = [r.get("emg_file", "") or "" for r in rows]

    order = np.argsort(frame_index)
    return {
        "timestamp": timestamp[order],
        "frame_index": frame_index[order],
        "angle": angle[order],
        "angle_timestamp": angle_timestamp[order],
        "frame_file": [frame_file[i] for i in order],
        "emg_file": [emg_file[i] for i in order],
    }


def _compute_emg_rms(rec_dir: Path, emg_files: list[str]) -> np.ndarray:
    rms = np.zeros((len(emg_files), EMG_CHANNELS), dtype=np.float32)
    emg_dir = rec_dir / "emg"
    for i, fname in enumerate(emg_files):
        if not fname:
            continue
        p = emg_dir / fname
        if not p.exists():
            continue
        arr = np.load(p)
        rms[i] = np.sqrt(np.mean(arr ** 2, axis=0))
    return rms


def load_recording(rec_dir: str | Path, compute_rms: bool = True) -> Recording:
    rec_dir = Path(rec_dir)
    if not rec_dir.is_dir():
        raise FileNotFoundError(f"Recording folder not found: {rec_dir}")

    meta = _load_metadata(find_csv(rec_dir))
    timestamps = meta["timestamp"]
    duration = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
    fps = (len(timestamps) - 1) / duration if duration > 0 else 0.0

    emg_rms = _compute_emg_rms(rec_dir, meta["emg_file"]) if compute_rms else np.zeros(
        (len(timestamps), EMG_CHANNELS), dtype=np.float32
    )

    return Recording(
        path=rec_dir,
        emg_rms=emg_rms,
        duration_s=duration,
        fps=fps,
        **meta,
    )


def print_summary(rec: Recording) -> None:
    print(f"\n{'=' * 50}")
    print(f"Recording: {rec.name}")
    print(f"Path:      {rec.path}")
    print(f"{'=' * 50}")
    print(f"Frames:    {rec.n_frames}")
    print(f"Duration:  {rec.duration_s:.2f} s")
    print(f"FPS (avg): {rec.fps:.2f}")

    valid_angle = ~np.isnan(rec.angle)
    if valid_angle.any():
        angles = rec.angle[valid_angle]
        print(f"\nAngle:")
        print(f"  min/max/mean: {angles.min():.0f} / {angles.max():.0f} / {angles.mean():.1f} deg")
        print(f"  unique values: {sorted({int(a) for a in angles})}")
        has_ts = ~np.isnan(rec.angle_timestamp)
        print(f"  angle_timestamp present: {has_ts.sum()}/{rec.n_frames}")

    active = rec.emg_rms.max(axis=1) > 0
    print(f"\nEMG:")
    print(f"  active frames: {active.sum()}/{rec.n_frames}")
    if active.any():
        first = int(np.argmax(active))
        print(f"  first active frame: #{first} ({first / max(rec.fps, 1):.1f}s)")
        for ch in range(EMG_CHANNELS):
            vals = rec.emg_rms[active, ch]
            print(f"  CH{ch + 1} RMS: min={vals.min():.4f}, max={vals.max():.4f}, mean={vals.mean():.4f} mV")

    frames_dir = rec.path / "frames"
    if frames_dir.is_dir():
        print(f"\nFiles:")
        print(f"  frames/: {len(list(frames_dir.glob('*.jpg')))} jpg")
        print(f"  emg/:    {len(list((rec.path / 'emg').glob('*.npy')))} npy")


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:
        raise SystemExit(
            "matplotlib failed to import (often caused by numpy version conflict).\n"
            "Try:  $env:PYTHONNOUSERSITE=1; python visualize_recording.py ...\n"
            f"Original error: {e}"
        ) from e


def plot_overview(rec: Recording, save_path: str | Path | None = None) -> None:
    import matplotlib
    if save_path:
        matplotlib.use("Agg")
    plt = _import_matplotlib()

    t = rec.timestamp - rec.timestamp[0]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [1, 1.2, 1.2]})
    fig.suptitle(
        f"{rec.name}  |  {rec.n_frames} frames, {rec.duration_s:.1f}s, ~{rec.fps:.1f} fps",
        fontsize=13,
    )

    ax = axes[0]
    valid = ~np.isnan(rec.angle)
    if valid.any():
        ax.plot(t[valid], rec.angle[valid], color="steelblue", linewidth=1.2)
        ax.set_ylabel("Angle (deg)")
    else:
        ax.text(0.5, 0.5, "No angle data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Servo Angle")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for ch in range(EMG_CHANNELS):
        ax.plot(t, rec.emg_rms[:, ch], color=EMG_CHANNEL_COLORS[ch], linewidth=0.8, label=f"CH{ch + 1}")
    ax.set_ylabel("EMG RMS (mV)")
    ax.set_title("EMG Envelope RMS (per frame, 100 ms window)")
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    emg_mean = rec.emg_rms.mean(axis=1)
    ax.fill_between(t, emg_mean, alpha=0.3, color="gray")
    ax.plot(t, emg_mean, color="black", linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean EMG RMS (mV)")
    ax.set_title("EMG Activity (channel average)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Overview saved -> {save_path}")
        plt.close(fig)
    else:
        plt.show()


def interactive_view(rec: Recording, start_frame: int = 0) -> None:
    plt = _import_matplotlib()
    from matplotlib.widgets import Slider

    n = rec.n_frames
    start_frame = int(np.clip(start_frame, 0, n - 1))
    t_rel = rec.timestamp - rec.timestamp[0]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"{rec.name} — interactive viewer  (←/→ or slider)", fontsize=13)

    ax_img = fig.add_axes([0.05, 0.55, 0.55, 0.38])
    ax_emg = fig.add_axes([0.05, 0.30, 0.90, 0.20])
    ax_timeline = fig.add_axes([0.05, 0.12, 0.90, 0.14])
    ax_slider = fig.add_axes([0.15, 0.03, 0.70, 0.03])
    ax_info = fig.add_axes([0.63, 0.55, 0.32, 0.38])
    ax_info.axis("off")

    img_artist = ax_img.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
    ax_img.set_title("Camera Frame")
    ax_img.axis("off")

    emg_t = np.arange(EMG_WINDOW_MS) / EMG_FS * 1000
    emg_lines = []
    for ch in range(EMG_CHANNELS):
        line, = ax_emg.plot(emg_t, np.zeros(EMG_WINDOW_MS), color=EMG_CHANNEL_COLORS[ch],
                            linewidth=1.2, label=f"CH{ch + 1}")
        emg_lines.append(line)
    ax_emg.set_xlabel("Time within window (ms)")
    ax_emg.set_ylabel("EMG envelope (mV)")
    ax_emg.set_title("EMG window at current frame (100 ms)")
    ax_emg.legend(loc="upper right", ncol=4, fontsize=8)
    ax_emg.grid(True, alpha=0.3)

    valid = ~np.isnan(rec.angle)
    if valid.any():
        ax_timeline.plot(t_rel[valid], rec.angle[valid], color="steelblue", linewidth=1, alpha=0.8)
        ax_timeline.set_ylabel("Angle (deg)", color="steelblue")
    ax_timeline_twin = ax_timeline.twinx()
    emg_mean = rec.emg_rms.mean(axis=1)
    ax_timeline_twin.plot(t_rel, emg_mean, color="orange", linewidth=0.8, alpha=0.7)
    ax_timeline_twin.set_ylabel("EMG RMS (mV)", color="orange")

    cursor_line = ax_timeline.axvline(t_rel[start_frame], color="red", linewidth=1.5, linestyle="--")
    ax_timeline.set_xlabel("Time (s)")
    ax_timeline.set_title("Timeline (red = current frame)")
    ax_timeline.grid(True, alpha=0.3)

    info_text = ax_info.text(0.05, 0.95, "", va="top", fontsize=10, family="monospace",
                             transform=ax_info.transAxes)
    slider = Slider(ax_slider, "Frame", 0, n - 1, valinit=start_frame, valstep=1)

    def render(frame_idx: int) -> None:
        frame_idx = int(frame_idx)

        fp = rec.frame_path(frame_idx)
        if fp is not None:
            img_artist.set_data(plt.imread(fp))
        else:
            img_artist.set_data(np.zeros((480, 640, 3), dtype=np.uint8))

        emg = rec.load_emg(frame_idx)
        if emg is not None:
            for ch, line in enumerate(emg_lines):
                line.set_ydata(emg[:, ch])
            ymax = max(float(emg.max()) * 1.15, 0.1)
            ax_emg.set_ylim(0, ymax)
        else:
            for line in emg_lines:
                line.set_ydata(np.zeros(EMG_WINDOW_MS))
            ax_emg.set_ylim(0, 1)

        cursor_line.set_xdata([t_rel[frame_idx], t_rel[frame_idx]])

        ts = datetime.fromtimestamp(rec.timestamp[frame_idx]).strftime("%H:%M:%S.%f")[:-3]
        angle = rec.get_angle(frame_idx)
        angle_str = f"{angle:.0f} deg" if angle is not None else "N/A"
        rms = rec.emg_rms[frame_idx]
        info_text.set_text(
            f"Frame:     #{frame_idx} / {n - 1}\n"
            f"Time:      {ts}\n"
            f"t_rel:     {t_rel[frame_idx]:.3f} s\n"
            f"Angle:     {angle_str}\n"
            f"EMG RMS:\n"
            f"  CH1 {rms[0]:.4f} mV\n"
            f"  CH2 {rms[1]:.4f} mV\n"
            f"  CH3 {rms[2]:.4f} mV\n"
            f"  CH4 {rms[3]:.4f} mV\n"
            f"  mean {rms.mean():.4f} mV"
        )
        fig.canvas.draw_idle()

    def on_slider(val):
        render(val)

    def on_key(event):
        cur = int(slider.val)
        if event.key in ("right", "up"):
            slider.set_val(min(cur + 1, n - 1))
        elif event.key in ("left", "down"):
            slider.set_val(max(cur - 1, 0))

    slider.on_changed(on_slider)
    fig.canvas.mpl_connect("key_press_event", on_key)
    render(start_frame)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Load and visualize multimodal recordings")
    parser.add_argument("--recording", "-r", type=str, default=None,
                        help="Path to recording_* folder (default: latest in project root)")
    parser.add_argument("--root", type=str, default=str(BASE_DIR),
                        help="Root directory to search for recordings")
    parser.add_argument("--list", action="store_true", help="List available recordings")
    parser.add_argument("--summary", action="store_true", help="Print summary only, no plot")
    parser.add_argument("--overview", action="store_true",
                        help="Show static overview plot instead of interactive viewer")
    parser.add_argument("--save", type=str, default=None, help="Save overview figure to path")
    parser.add_argument("--frame", type=int, default=0, help="Start frame for interactive viewer")
    args = parser.parse_args()

    root = Path(args.root)
    recordings = find_recordings(root)

    if args.list:
        if not recordings:
            print(f"No recording_* folders under {root}")
            return
        print(f"Found {len(recordings)} recording(s) under {root}:\n")
        for p in recordings:
            try:
                csv_path = find_csv(p)
                n_frames = sum(1 for _ in open(csv_path, encoding="utf-8")) - 1
            except FileNotFoundError:
                n_frames = "?"
            print(f"  {p.name}  ({n_frames} frames)")
        return

    if args.recording:
        rec_path = Path(args.recording)
        if not rec_path.is_absolute():
            rec_path = root / rec_path
    elif recordings:
        rec_path = recordings[-1]
        print(f"Using latest recording: {rec_path.name}")
    else:
        parser.error("No recording found. Use --recording or --list")

    print(f"Loading {rec_path.name} ...")
    rec = load_recording(rec_path)
    print_summary(rec)

    if args.summary and not args.overview and not args.save:
        return

    if args.overview or args.save:
        plot_overview(rec, save_path=args.save)
    else:
        interactive_view(rec, start_frame=args.frame)


if __name__ == "__main__":
    main()
