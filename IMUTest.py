






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
CSV_COLUMNS += ['world_ax', 'world_ay', 'world_az', 'forward_raw', 'forward_ema',
                'forward_3hz', 'forward_5hz', 'forward_x', 'forward_y', 'forward_z', 'direction_id']


def body_to_world(axes, angles):
    roll, pitch, yaw = np.deg2rad(angles)
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    rotation = np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                         [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                         [-sp, cp*sr, cp*cr]])
    return rotation @ axes


class LiveAccelerationFilters:





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
        self.world_filters = LiveAccelerationFilters()
        self.forward = None
        self.direction_id = 0
        self.direction_pending = False
        self.direction_samples = []
        self.direction_wait_samples = 0
        self.signed = np.full(4, np.nan)

    def learn_forward(self):
        with self.lock:
            if self.calibrating:
                return
            self.forward = None
            self.signed[:] = np.nan
            self.direction_pending = True
            self.direction_samples = []
            self.direction_wait_samples = 0
            self.status = 'Direction: start still, then move straight toward the object. Do not move back first.'

    def update_direction(self, world):
        if not self.direction_pending:
            return
        self.direction_wait_samples += 1
        horizontal = world.copy()
        horizontal[2] = 0
        if self.direction_samples or np.linalg.norm(horizontal) >= .35:
            self.direction_samples.append(horizontal)
        if len(self.direction_samples) >= 6:
            samples = np.array(self.direction_samples)
            mean = samples.mean(axis=0)
            strength = np.linalg.norm(mean)
            consistency = strength / max(np.linalg.norm(samples, axis=1).mean(), 1e-9)
            self.direction_pending = False
            if strength < .2 or consistency < .75:
                self.status = 'Direction not stable. Press D, start still and make one straight forward movement.'
            else:
                self.forward = mean / strength
                self.direction_id += 1
                self.points.clear()
                self.origin = None
                self.status = 'Direction ready: toward object is positive. R: recalibrate; D: relearn direction.'
                print('[IMU] Forward direction in world coordinates:', self.forward)
        elif self.direction_wait_samples > IMU_RATE_HZ * 8:
            self.direction_pending = False
            self.status = 'No clear forward movement detected. Press D to retry.'

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
            self.world_filters.reset()
            self.forward = None
            self.direction_pending = False
            self.signed[:] = np.nan
            self.status = 'Keep IMU still: calibrating for 2 seconds...'

    def calibration_quality(self):





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
            self.world_filters.reset()
            self.forward = None
            self.direction_pending = False
            self.signed[:] = np.nan
            self.status = 'Calibrated. Press D while still, then move straight toward the object to learn direction.'
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
            world = body_to_world(self.linear, angles)
            world_filtered = self.world_filters.update(world)
            self.update_direction(world)
            if self.forward is not None:
                self.signed = np.array([np.dot(a, self.forward) for a in [world, *world_filtered]])
            if self.origin is None:
                self.origin = timestamp
            elapsed = timestamp - self.origin
            self.points.append((elapsed, self.linear_mag, *self.filtered_mags, *self.signed))
            self.rows.append([len(self.rows), timestamp, *self.latest, self.mag, self.calibration_id,
                              *raw, *gyro, *angles, *gravity, *self.linear, self.linear_mag,
                              *[v for axes in filtered for v in (*axes, float(np.linalg.norm(axes)))],
                              *world, *self.signed, *(self.forward if self.forward is not None else np.full(3, np.nan)),
                              self.direction_id if self.forward is not None else 0])

    def notify(self, sender, data):
        if self.stop.is_set():
            return
        received = time.time()
                                                                    
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
    window.title('IMU — Magnitude and signed forward acceleration')
    window.geometry('1150x850')
    status, values = tk.StringVar(value=state.status), tk.StringVar(value='Waiting for IMU...')
    tk.Label(window, textvariable=status, font=('Segoe UI', 12)).pack(pady=8)
    tk.Label(window, textvariable=values, font=('Segoe UI', 11)).pack()
    fig = Figure(figsize=(10, 5), dpi=100)
    ax = fig.add_subplot(211)
    signed_ax = fig.add_subplot(212)
    plot_lines = [
        ax.plot([], [], color='gray', alpha=.45, lw=1, label='Gravity removed: unsmoothed')[0],
        ax.plot([], [], color='tab:blue', lw=1.4, label='EMA (same time constant as current smoothing)')[0],
        ax.plot([], [], color='tab:orange', lw=2, label='Causal low-pass 3 Hz')[0],
        ax.plot([], [], color='tab:green', lw=1.7, label='Causal low-pass 5 Hz')[0],
    ]
    ax.set(xlabel='Time since calibration (s)', ylabel='Acceleration magnitude (m/s²)', xlim=(0, PLOT_SECONDS), ylim=(0, 1))
    ax.grid(alpha=.25)
    ax.legend(loc='upper left')
    signed_lines = [signed_ax.plot([], [], color=color, lw=width, alpha=alpha, label=label)[0]
                    for color, width, alpha, label in [('gray', 1, .45, 'Signed raw'),
                    ('tab:blue', 1.4, 1, 'Signed EMA'), ('tab:orange', 2, 1, 'Signed 3 Hz'),
                    ('tab:green', 1.7, 1, 'Signed 5 Hz')]]
    signed_ax.axhline(0, color='black', lw=.8)
    signed_ax.set(xlabel='Time since direction setup (s)', ylabel='Forward acceleration (m/s²)',
                  xlim=(0, PLOT_SECONDS), ylim=(-1, 1), title='Fixed horizontal forward axis: toward object = positive')
    signed_ax.grid(alpha=.25)
    signed_ax.legend(loc='upper left')
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

    def direction(event=None):
        if not closing:
            state.learn_forward()

    controls = tk.Frame(window)
    controls.pack(pady=8)
    visibility = []
    for plot_line, label in zip(plot_lines, ['Raw', 'EMA', '3 Hz', '5 Hz']):
        checked = tk.BooleanVar(value=True)
        visibility.append(checked)
        tk.Checkbutton(controls, text=label, variable=checked).pack(side=tk.LEFT, padx=4)
    tk.Button(controls, text='Recalibrate (R)', command=recalibrate).pack(side=tk.LEFT, padx=10)
    tk.Button(controls, text='Learn forward direction (D)', command=direction).pack(side=tk.LEFT, padx=10)
    tk.Button(controls, text='Save and exit (Q)', command=close).pack(side=tk.LEFT, padx=10)
    for key in ['r', 'R']:
        window.bind('<'+key+'>', recalibrate)
    for key in ['q', 'Q']:
        window.bind('<'+key+'>', close)
    for key in ['d', 'D']:
        window.bind('<'+key+'>', direction)
    window.protocol('WM_DELETE_WINDOW', close)

    def update():
        if closing:
            return
        with state.lock:
            points, text = list(state.points), state.status
            acceleration, mag, count = state.linear.copy(), state.mag, len(state.rows)
            linear_mag, angles = state.linear_mag, state.angles.copy()
            filtered_mags = state.filtered_mags.copy()
            signed_values = state.signed.copy()
            if state.calibrating and state.started is not None:
                remaining = CALIB_SECONDS - (time.monotonic() - state.started)
                if remaining > 0:
                    text += f' {remaining:.1f} s remaining'
        status.set(text)
        signed_text = f'{signed_values[2]:+.3f}' if np.isfinite(signed_values[2]) else 'direction not set'
        values.set(f'Magnitude 3 Hz={filtered_mags[1]:.3f} | Signed 3 Hz={signed_text} m/s² | samples={count}')
        if points:
            latest = max(p[0] for p in points)
            start = max(0, latest - PLOT_SECONDS)
            visible = np.array([p for p in points if p[0] >= start])
            for i, (plot_line, checked) in enumerate(zip(plot_lines, visibility), 1):
                plot_line.set_data(visible[:, 0], visible[:, i])
                plot_line.set_visible(checked.get())
                signed_lines[i-1].set_data(visible[:, 0], visible[:, i+4])
                signed_lines[i-1].set_visible(checked.get())
            ax.set_xlim(start, max(PLOT_SECONDS, latest))
            selected = [i + 1 for i, checked in enumerate(visibility) if checked.get()]
            ax.set_ylim(0, max(.3, float(visible[:, selected].max()) * 1.15) if selected else .3)
            signed_ax.set_xlim(start, max(PLOT_SECONDS, latest))
            signed_data = visible[:, [i+4 for i in selected]] if selected else np.array([])
            finite_signed = signed_data[np.isfinite(signed_data)]
            limit = max(.3, float(np.max(np.abs(finite_signed))) * 1.15) if finite_signed.size else 1
            signed_ax.set_ylim(-limit, limit)
            canvas.draw_idle()
        else:
            for plot_line in [*plot_lines, *signed_lines]:
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
    for angles in [(0,0,0),(30,20,70),(-45,10,120)]:
        rotation = np.column_stack([body_to_world(np.eye(3)[:,j], angles) for j in range(3)])
        np.testing.assert_allclose(body_to_world(rotation.T @ np.array([1.,-.5,.2]), angles), [1.,-.5,.2], atol=1e-12)
    state = State()
    state.calibrating = False
    state.learn_forward()
    for _ in range(6):
        state.update_direction(np.array([.6,.8,0.]))
    np.testing.assert_allclose(state.forward, [.6,.8,0.])
    assert np.dot(state.forward, [.6,.8,0.]) > 0
    assert np.dot(state.forward, [-.6,-.8,0.]) < 0
    state.process(sample([.1,0,1]), 200)
    assert len(state.rows[-1]) == len(CSV_COLUMNS)
    state.begin()
    assert state.forward is None and np.isnan(state.signed).all()
    print('PASS: world rotation, forward direction, signed projection, CSV columns and reset')
    print('PASS: parsing, tilted/inverted static poses, yaw invariance, dynamic acceleration, bias and recalibration')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-time signed acceleration test')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main()
