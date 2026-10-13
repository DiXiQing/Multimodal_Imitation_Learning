"""Compare real-time acceleration smoothing after gravity compensation.

R: recalibrate while stationary. Q / close: save CSV and quit.
Stable two-second calibration; causal 3/5 Hz low-pass at nominal 50 Hz.
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
from scipy.signal import butter, sosfilt, sosfilt_zi
from bleak import BleakClient
from imu_processing import gravity_in_sensor, parse_imu_packet, linear_acceleration, calibration_quality

DEVICE_ADDRESS = 'D8:76:A8:5D:CC:CF'
NOTIFY_UUID = '0000ffe4-0000-1000-8000-00805f9a34fb'
CALIB_SECONDS = 2.0
CALIB_MAX_WAIT_SECONDS = 20.0
GRAVITY = 9.81
PLOT_SECONDS = 20.0
IMU_RATE_HZ = 50.0
# Same time constant as span=5 EMA at 30 Hz, applied at the IMU rate.
EMA_ALPHA = 1 - (1 - 1 / 3) ** (30.0 / IMU_RATE_HZ)
OUTPUT_DIR = Path(__file__).resolve().parent / 'Data' / 'BackupData' / 'IMU_Acceleration_Test'
CSV_COLUMNS = ['imu_index', 't_imu', 'ax', 'ay', 'az', 'mag', 'calibration_id',
               'ax_raw_g', 'ay_raw_g', 'az_raw_g', 'gx_dps', 'gy_dps', 'gz_dps',
               'roll_deg', 'pitch_deg', 'yaw_deg',
               'gravity_x_g', 'gravity_y_g', 'gravity_z_g',
               'ax_linear', 'ay_linear', 'az_linear', 'mag_linear',
               'ax_ema', 'ay_ema', 'az_ema', 'mag_ema',
               'ax_lowpass_3hz', 'ay_lowpass_3hz', 'az_lowpass_3hz', 'mag_lowpass_3hz',
               'ax_lowpass_5hz', 'ay_lowpass_5hz', 'az_lowpass_5hz', 'mag_lowpass_5hz']


class LiveAccelerationFilters:
    """One update per hardware sample; never use BLE arrival jitter as dt.

    All axes use identical causal filters before their vector norm is computed.
    Reset on calibration/reconnect to avoid carrying old filter history.
    """
    def __init__(self):
        self.sos = {hz: butter(2, hz, fs=IMU_RATE_HZ, output='sos') for hz in (3, 5)}
        self.reset()

    def reset(self):
        self.ema = None
        self.history = {}

    def update(self, axes):
        axes = np.asarray(axes, dtype=float)
        if self.ema is None:
            self.ema = axes.copy()
            self.history = {hz: sosfilt_zi(sos)[:, :, None] * axes[None, None, :]
                            for hz, sos in self.sos.items()}
        else:
            self.ema += EMA_ALPHA * (axes - self.ema)
        result = [self.ema.copy()]
        for hz, sos in self.sos.items():
            value, self.history[hz] = sosfilt(sos, axes[None, :], axis=0, zi=self.history[hz])
            result.append(value[0])
        return result


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
        self.calibration_times = []
        self.gyro_samples = []
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
        self.filters = LiveAccelerationFilters()
        self.filtered_mags = np.zeros(3)

    def begin(self):
        with self.lock:
            self.calibrating = True
            self.started = time.monotonic()
            self.samples = []
            self.linear_samples = []
            self.calibration_times = []
            self.gyro_samples = []
            self.filters.reset()
            self.filtered_mags[:] = 0
            self.points.clear()
            self.status = 'Keep IMU still: calibrating for 2 seconds...'

    def calibration_quality(self):
        """Called with lock held; reject transient attitude or moving samples.

        A single static pose cannot identify arbitrary angle offsets and sensor
        bias separately. Reject a large unexplained residual instead of hiding it.
        """
        return calibration_quality(self.samples, self.linear_samples,
                                   self.gyro_samples, self.calibration_times)

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
            self.filters.reset()
            self.filtered_mags[:] = 0
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
                self.calibration_times.append(timestamp)
                self.gyro_samples.append(gyro.copy())
                # Only the latest stationary interval is used after warm-up.
                while len(self.calibration_times)>1 and timestamp-self.calibration_times[0]>2.1:
                    self.calibration_times.pop(0)
                    self.samples.pop(0)
                    self.linear_samples.pop(0)
                    self.gyro_samples.pop(0)
                return
            self.latest = (raw - self.bias) * GRAVITY
            self.mag = float(np.linalg.norm(self.latest))
            self.linear = linear_acceleration(packet, self.linear_bias, GRAVITY)
            self.linear_mag = float(np.linalg.norm(self.linear))
            filtered = self.filters.update(self.linear)
            self.filtered_mags = np.array([np.linalg.norm(a) for a in filtered])
            if self.origin is None:
                self.origin = timestamp
            elapsed = timestamp - self.origin
            self.points.append((elapsed, self.linear_mag, *self.filtered_mags))
            self.rows.append([len(self.rows), timestamp, *self.latest, self.mag, self.calibration_id,
                              *raw, *gyro, *angles, *gravity, *self.linear, self.linear_mag,
                              *[v for axes in filtered for v in (*axes, float(np.linalg.norm(axes)))]])

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
                            with state.lock:
                                good, reason = state.calibration_quality()
                                waited = time.monotonic()-state.started
                                if not good:
                                    state.status = reason + ' | calibration pending'
                            if good:
                                state.finish()
                            elif waited > CALIB_MAX_WAIT_SECONDS:
                                raise RuntimeError('Calibration rejected: '+reason+'. Keep still or check sensor angle calibration.')
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
    window.title('IMU — Real-time smoothing: EMA / 3 Hz / 5 Hz')
    window.geometry('1050x650')
    status, values = tk.StringVar(value=state.status), tk.StringVar(value='Waiting for IMU...')
    tk.Label(window, textvariable=status, font=('Segoe UI', 12)).pack(pady=8)
    tk.Label(window, textvariable=values, font=('Segoe UI', 11)).pack()
    fig = Figure(figsize=(10, 5), dpi=100)
    ax = fig.add_subplot(111)
    plot_lines = [
        ax.plot([], [], color='gray', alpha=.45, lw=1, label='Gravity removed: unsmoothed')[0],
        ax.plot([], [], color='tab:blue', lw=1.4, label='EMA (same time constant as current smoothing)')[0],
        ax.plot([], [], color='tab:orange', lw=2, label='Causal low-pass 3 Hz')[0],
        ax.plot([], [], color='tab:green', lw=1.7, label='Causal low-pass 5 Hz')[0],
    ]
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
    visibility = []
    for plot_line, label in zip(plot_lines, ['Raw', 'EMA', '3 Hz', '5 Hz']):
        checked = tk.BooleanVar(value=True)
        visibility.append(checked)
        tk.Checkbutton(controls, text=label, variable=checked).pack(side=tk.LEFT, padx=4)
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
            filtered_mags = state.filtered_mags.copy()
            if state.calibrating and state.started is not None:
                remaining = CALIB_SECONDS - (time.monotonic() - state.started)
                if remaining > 0:
                    text += f' {remaining:.1f} s remaining'
        status.set(text)
        values.set(f'Raw={linear_mag:.3f} | EMA={filtered_mags[0]:.3f} | 3 Hz={filtered_mags[1]:.3f} | 5 Hz={filtered_mags[2]:.3f} m/s² | samples={count}')
        if points:
            latest = max(p[0] for p in points)
            start = max(0, latest - PLOT_SECONDS)
            visible = np.array([p for p in points if p[0] >= start])
            for i, (plot_line, checked) in enumerate(zip(plot_lines, visibility), 1):
                plot_line.set_data(visible[:, 0], visible[:, i])
                plot_line.set_visible(checked.get())
            ax.set_xlim(start, max(PLOT_SECONDS, latest))
            selected = [i + 1 for i, checked in enumerate(visibility) if checked.get()]
            ax.set_ylim(0, max(.3, float(visible[:, selected].max()) * 1.15) if selected else .3)
            canvas.draw_idle()
        else:
            for plot_line in plot_lines:
                plot_line.set_data([], [])
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
    # DC must be preserved, and 3 Hz must suppress 10 Hz more than 5 Hz.
    filters = LiveAccelerationFilters()
    for _ in range(100):
        values = filters.update([.4, -.2, .1])
        for value in values:
            np.testing.assert_allclose(value, [.4, -.2, .1], atol=1e-10)
    filters.reset()
    waves = []
    for i in range(500):
        values = filters.update([np.sin(2*np.pi*10*i/IMU_RATE_HZ), 0, 0])
        waves.append([v[0] for v in values])
    rms = np.sqrt(np.mean(np.array(waves)[100:]**2, axis=0))
    assert rms[1] < rms[2] < .25, rms
    filters.reset()
    for value in filters.update([0,0,0]):
        np.testing.assert_array_equal(value, np.zeros(3))
    print('PASS: causal filters preserve DC, suppress high frequencies and reset after calibration')
    print('PASS: parsing, tilted/inverted static poses, yaw invariance, dynamic acceleration, bias and recalibration')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main()
