"""Live IMU magnitude: same two-second fixed-pose calibration as Record.

R: recalibrate while stationary. Q / close: save CSV and quit.
The magnitude is calculated from corrected axes without extra smoothing.
"""
import argparse
import asyncio
import csv
from collections import deque
from datetime import datetime
from pathlib import Path
import struct
import threading
import time
import numpy as np
from bleak import BleakClient

DEVICE_ADDRESS = 'D8:76:A8:5D:CC:CF'
NOTIFY_UUID = '0000ffe4-0000-1000-8000-00805f9a34fb'
CALIB_SECONDS = 2.0
GRAVITY = 9.81
PLOT_SECONDS = 20.0
OUTPUT_DIR = Path(__file__).resolve().parent / 'Data' / 'BackupData' / 'IMU_Acceleration_Test'


def extract_imu_packets(data):
    packets, i = [], 0
    while i <= len(data) - 20:
        if data[i:i+2] == b'\x55\x61':
            values = struct.unpack('<9h', data[i+2:i+20])
            packets.append(np.array(values[:3], dtype=float) / 32768 * 16)
            i += 20
        else:
            i += 1
    return packets


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.recalibrate = threading.Event()
        self.status = 'Connecting to IMU...'
        self.calibrating = True
        self.started = None
        self.samples = []
        self.bias = np.zeros(3)
        self.latest = np.zeros(3)
        self.mag = 0.
        self.origin = None
        self.points = deque(maxlen=10000)
        self.rows = []
        self.calibration_id = 0

    def begin(self):
        with self.lock:
            self.calibrating = True
            self.started = time.monotonic()
            self.samples = []
            self.status = 'Keep IMU still: calibrating for 2 seconds...'

    def finish(self):
        with self.lock:
            if not self.samples:
                raise RuntimeError('No IMU samples received during calibration')
            self.bias = np.mean(self.samples, axis=0)
            self.calibrating = False
            self.calibration_id += 1
            self.points.clear()
            self.origin = None
            self.latest = np.zeros(3)
            self.mag = 0.
            self.status = 'Live | R: recalibrate | Q: save and exit'
            print('[IMU] Calibration complete; bias (g):', self.bias)

    def process(self, raw, timestamp):
        with self.lock:
            if self.calibrating:
                self.samples.append(raw.copy())
                return
            self.latest = (raw - self.bias) * GRAVITY
            self.mag = float(np.linalg.norm(self.latest))
            if self.origin is None:
                self.origin = timestamp
            elapsed = timestamp - self.origin
            self.points.append((elapsed, self.mag))
            self.rows.append([len(self.rows), timestamp, *self.latest, self.mag, self.calibration_id])

    def notify(self, sender, data):
        if self.stop.is_set():
            return
        received = time.time()
        # Same 50 Hz batch timestamp convention as Record_mediapipe.
        for i, raw in enumerate(extract_imu_packets(bytes(data))):
            self.process(raw, received + i / 50.)


async def measure(state):
    for attempt in range(1, 4):
        if state.stop.is_set():
            return
        try:
            with state.lock:
                state.status = f'Connecting to IMU ({attempt}/3)...'
                state.calibrating = True
            async with BleakClient(DEVICE_ADDRESS, timeout=20.) as client:
                state.begin()
                await client.start_notify(NOTIFY_UUID, state.notify)
                try:
                    while not state.stop.is_set():
                        if not client.is_connected:
                            raise ConnectionError('IMU disconnected')
                        if state.recalibrate.is_set():
                            state.recalibrate.clear()
                            state.begin()
                        with state.lock:
                            ready = state.calibrating and time.monotonic() - state.started >= CALIB_SECONDS
                        if ready:
                            state.finish()
                        await asyncio.sleep(.05)
                finally:
                    if client.is_connected:
                        await client.stop_notify(NOTIFY_UUID)
            return
        except Exception as exc:
            print(f'[IMU] Attempt {attempt}: {exc}')
            with state.lock:
                state.status = f'IMU error: {exc}'
                state.calibrating = True
                state.started = None
            if attempt < 3:
                for _ in range(40):
                    if state.stop.is_set():
                        return
                    await asyncio.sleep(.05)
            else:
                with state.lock:
                    state.status = 'Connection failed. Close the window to exit.'


def worker(state):
    try:
        asyncio.run(measure(state))
    finally:
        if state.rows:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            path = OUTPUT_DIR / f'acceleration_test_{datetime.now():%Y%m%d_%H%M%S_%f}.csv'
            with path.open('w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['imu_index', 't_imu', 'ax', 'ay', 'az', 'mag', 'calibration_id'])
                writer.writerows(state.rows)
            print('[Saved]', path)


def main():
    import tkinter as tk
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    state = State()
    window = tk.Tk()
    window.title('IMU — Real-time acceleration magnitude')
    window.geometry('1050x650')
    status, values = tk.StringVar(value=state.status), tk.StringVar(value='Waiting for IMU...')
    tk.Label(window, textvariable=status, font=('Segoe UI', 12)).pack(pady=8)
    tk.Label(window, textvariable=values, font=('Segoe UI', 11)).pack()
    fig = Figure(figsize=(10, 5), dpi=100)
    ax = fig.add_subplot(111)
    line, = ax.plot([], [], color='tab:blue', lw=1.5)
    ax.set(xlabel='Time since calibration (s)', ylabel='Acceleration magnitude (m/s²)', xlim=(0, PLOT_SECONDS), ylim=(0, 1))
    ax.grid(alpha=.25)
    fig.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    thread = threading.Thread(target=worker, args=(state,), daemon=True)
    closing = False

    def shutdown():
        if thread.is_alive():
            window.after(50, shutdown)
        else:
            window.destroy()

    def close(event=None):
        nonlocal closing
        if not closing:
            closing = True
            state.stop.set()
            status.set('Stopping BLE and saving CSV...')
            window.after(50, shutdown)

    def recalibrate(event=None):
        if not closing:
            state.recalibrate.set()

    controls = tk.Frame(window)
    controls.pack(pady=8)
    tk.Button(controls, text='Recalibrate (R)', command=recalibrate).pack(side=tk.LEFT, padx=10)
    tk.Button(controls, text='Save and exit (Q)', command=close).pack(side=tk.LEFT, padx=10)
    for key in ['r', 'R']:
        window.bind('<'+key+'>', recalibrate)
    for key in ['q', 'Q']:
        window.bind('<'+key+'>', close)
    window.protocol('WM_DELETE_WINDOW', close)

    def update():
        if closing:
            return
        with state.lock:
            points, text = list(state.points), state.status
            acceleration, mag, count = state.latest.copy(), state.mag, len(state.rows)
            if state.calibrating and state.started is not None:
                remaining = max(0, CALIB_SECONDS - (time.monotonic() - state.started))
                text += f' {remaining:.1f} s remaining'
        status.set(text)
        values.set(f'ax={acceleration[0]:.3f}  ay={acceleration[1]:.3f}  az={acceleration[2]:.3f} m/s²    magnitude={mag:.3f} m/s²    samples={count}')
        if points:
            latest = max(t for t, a in points)
            start = max(0, latest - PLOT_SECONDS)
            visible = [(t, a) for t, a in points if t >= start]
            line.set_data(*zip(*visible))
            ax.set_xlim(start, max(PLOT_SECONDS, latest))
            ax.set_ylim(0, max(.3, max(a for t, a in visible) * 1.15))
            canvas.draw_idle()
        else:
            line.set_data([], [])
            canvas.draw_idle()
        window.after(50, update)

    thread.start()
    update()
    try:
        window.mainloop()
    finally:
        state.stop.set()
        thread.join()


def self_test():
    packet = b'\x55\x61' + struct.pack('<9h', 2048, -4096, 1024, 0, 0, 0, 0, 0, 0)
    samples = extract_imu_packets(b'noise' + packet + packet)
    assert len(samples) == 2
    np.testing.assert_allclose(samples[0], [1, -2, .5])
    state = State();state.begin()
    state.process(np.array([0., 0., 1.]), 100)
    assert not state.rows
    state.finish()
    state.process(np.array([0., 0., 1.]), 101)
    assert state.mag == 0
    state.process(np.array([.1, -.2, 1.]), 101.02)
    np.testing.assert_allclose(state.latest, [.981, -1.962, 0])
    np.testing.assert_allclose(state.mag, np.linalg.norm(state.latest))
    state.begin();state.process(np.array([1., 0., 0.]), 102);state.finish()
    state.process(np.array([1., 0., 0.]), 103)
    assert state.mag == 0 and state.calibration_id == 2
    print('PASS: parsing, static calibration, magnitude and recalibration')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main()
