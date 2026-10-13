"""Compare fixed-pose calibration with attitude-based gravity compensation.

R: recalibrate while stationary. Q / close: save CSV and quit.
Two-second calibration. Both curves use unsmoothed sensor measurements.
Protocol: WT9011DCL 0x61, Euler body-to-world Rz(yaw) Ry(pitch) Rx(roll).
https://wit-motion.gitbook.io/witmotion-sdk/wit-standard-protocol/wit-standard-communication-protocol
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
from imu_processing import gravity_in_sensor, parse_imu_packet, linear_acceleration

DEVICE_ADDRESS = 'D8:76:A8:5D:CC:CF'
NOTIFY_UUID = '0000ffe4-0000-1000-8000-00805f9a34fb'
CALIB_SECONDS = 2.0
GRAVITY = 9.81
PLOT_SECONDS = 20.0
OUTPUT_DIR = Path(__file__).resolve().parent / 'Data' / 'BackupData' / 'IMU_Acceleration_Test'
CSV_COLUMNS = ['imu_index', 't_imu', 'ax', 'ay', 'az', 'mag', 'calibration_id',
               'ax_raw_g', 'ay_raw_g', 'az_raw_g', 'gx_dps', 'gy_dps', 'gz_dps',
               'roll_deg', 'pitch_deg', 'yaw_deg',
               'gravity_x_g', 'gravity_y_g', 'gravity_z_g',
               'ax_linear', 'ay_linear', 'az_linear', 'mag_linear']


def extract_imu_packets(data):
    packets, i = [], 0
    while i <= len(data) - 20:
        if data[i:i+2] == b'\x55\x61':
            packets.append(parse_imu_packet(data[i:i+20]))
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
        self.linear_samples = []
        self.bias = np.zeros(3)
        self.linear_bias = np.zeros(3)
        self.latest = np.zeros(3)
        self.linear = np.zeros(3)
        self.linear_mag = 0.
        self.angles = np.zeros(3)
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
            self.linear_samples = []
            self.status = 'Keep IMU still: calibrating for 2 seconds...'

    def finish(self):
        with self.lock:
            if not self.samples:
                raise RuntimeError('No IMU samples received during calibration')
            self.bias = np.mean(self.samples, axis=0)
            self.linear_bias = np.mean(self.linear_samples, axis=0)
            self.calibrating = False
            self.calibration_id += 1
            self.points.clear()
            self.origin = None
            self.latest = np.zeros(3)
            self.mag = 0.
            self.linear = np.zeros(3)
            self.linear_mag = 0.
            self.status = 'Live | R: recalibrate | Q: save and exit'
            print('[IMU] Calibration complete; bias (g):', self.bias)
            print('[IMU] After gravity removal, sensor offset (g):', self.linear_bias)
            if np.linalg.norm(self.linear_bias) > .15:
                self.status = 'Large calibration residual: check static pose / sensor angle calibration | R: retry'
                print('[IMU] Warning: large gravity-model residual; verify static tilt before trusting correction.')

    def process(self, packet, timestamp):
        raw, gyro, angles = packet['acc'], packet['gyro'], packet['angles']
        gravity = gravity_in_sensor(angles)
        with self.lock:
            self.angles = angles.copy()
            if self.calibrating:
                self.samples.append(raw.copy())
                self.linear_samples.append(raw - gravity)
                return
            self.latest = (raw - self.bias) * GRAVITY
            self.mag = float(np.linalg.norm(self.latest))
            self.linear = linear_acceleration(packet, self.linear_bias, GRAVITY)
            self.linear_mag = float(np.linalg.norm(self.linear))
            if self.origin is None:
                self.origin = timestamp
            elapsed = timestamp - self.origin
            self.points.append((elapsed, self.mag, self.linear_mag))
            self.rows.append([len(self.rows), timestamp, *self.latest, self.mag, self.calibration_id,
                              *raw, *gyro, *angles, *gravity, *self.linear, self.linear_mag])

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
                writer.writerow(CSV_COLUMNS)
                writer.writerows(state.rows)
            print('[Saved]', path)


def main():
    import tkinter as tk
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    state = State()
    window = tk.Tk()
    window.title('IMU — Gravity compensation comparison')
    window.geometry('1050x650')
    status, values = tk.StringVar(value=state.status), tk.StringVar(value='Waiting for IMU...')
    tk.Label(window, textvariable=status, font=('Segoe UI', 12)).pack(pady=8)
    tk.Label(window, textvariable=values, font=('Segoe UI', 11)).pack()
    fig = Figure(figsize=(10, 5), dpi=100)
    ax = fig.add_subplot(111)
    line, = ax.plot([], [], color='tab:orange', lw=1.5, label='Original: fixed-pose calibration')
    linear_line, = ax.plot([], [], color='tab:blue', lw=1.5, label='Attitude-based gravity removal')
    ax.set(xlabel='Time since calibration (s)', ylabel='Acceleration magnitude (m/s²)', xlim=(0, PLOT_SECONDS), ylim=(0, 1))
    ax.grid(alpha=.25)
    ax.legend(loc='upper left')
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
            acceleration, mag, count = state.linear.copy(), state.mag, len(state.rows)
            linear_mag, angles = state.linear_mag, state.angles.copy()
            if state.calibrating and state.started is not None:
                remaining = max(0, CALIB_SECONDS - (time.monotonic() - state.started))
                text += f' {remaining:.1f} s remaining'
        status.set(text)
        values.set(f'Original={mag:.3f} | Linear={linear_mag:.3f} m/s² | Roll={angles[0]:.1f}° Pitch={angles[1]:.1f}° Yaw={angles[2]:.1f}° | samples={count}')
        if points:
            latest = max(p[0] for p in points)
            start = max(0, latest - PLOT_SECONDS)
            visible = np.array([p for p in points if p[0] >= start])
            line.set_data(visible[:, 0], visible[:, 1])
            linear_line.set_data(visible[:, 0], visible[:, 2])
            ax.set_xlim(start, max(PLOT_SECONDS, latest))
            ax.set_ylim(0, max(.3, float(visible[:, 1:].max()) * 1.15))
            canvas.draw_idle()
        else:
            line.set_data([], [])
            linear_line.set_data([], [])
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
    packet = b'\x55\x61' + struct.pack('<9h', 2048, -4096, 1024, 16384, -8192, 0, 8192, -4096, 16384)
    samples = extract_imu_packets(b'noise' + packet + packet)
    assert len(samples) == 2
    np.testing.assert_allclose(samples[0]['acc'], [1, -2, .5])
    np.testing.assert_allclose(samples[0]['gyro'], [1000, -500, 0])
    np.testing.assert_allclose(samples[0]['angles'], [45, -22.5, 90])
    def sample(acc, angles=(0, 0, 0)):
        return dict(acc=np.array(acc, dtype=float), gyro=np.zeros(3), angles=np.array(angles, dtype=float))
    state = State();state.begin()
    offset = np.array([.01, -.02, .015])
    state.process(sample(np.array([0., 0., 1.]) + offset), 100)
    assert not state.rows
    state.finish()
    state.process(sample(np.array([0., 0., 1.]) + offset), 101)
    assert state.mag == 0
    state.process(sample(np.array([.1, -.2, 1.]) + offset), 101.02)
    np.testing.assert_allclose(state.latest, [.981, -1.962, 0])
    np.testing.assert_allclose(state.mag, np.linalg.norm(state.latest))
    # Independent rotations: body-to-world Rz Ry Rx, rotate world +Z into body.
    for k, angles in enumerate([(30,0,0),(0,45,0),(40,-35,120),(180,0,0),(0,90,60),(0,0,150)]):
        r,p,y=np.deg2rad(angles)
        rx=np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
        ry=np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
        rz=np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]])
        gravity=(rz@ry@rx).T@np.array([0.,0.,1.])
        np.testing.assert_allclose(gravity_in_sensor(angles),gravity,atol=1e-12)
        state.process(sample(gravity+offset,angles),102+k)
        assert state.linear_mag < 1e-10
        dynamic=np.array([.12,-.07,.03])
        state.process(sample(gravity+offset+dynamic,angles),102.02+k)
        np.testing.assert_allclose(state.linear,dynamic*GRAVITY,atol=1e-12)
    state.begin();state.process(sample(np.array([1.,0.,0.])+offset,(0,-90,0)),110);state.finish()
    state.process(sample(np.array([1.,0.,0.])+offset,(0,-90,0)),111)
    assert state.mag == 0 and state.calibration_id == 2
    assert state.linear_mag < 1e-10
    assert len(state.rows[-1]) == len(CSV_COLUMNS)
    print('PASS: parsing, tilted/inverted static poses, yaw invariance, dynamic acceleration, bias and recalibration')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main()
