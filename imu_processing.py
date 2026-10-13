"""Shared WT9011DCL decoding and body-frame gravity compensation.

WIT ENU / Z-Y-X Euler angles: R = Rz(yaw) Ry(pitch) Rx(roll).
The accelerometer's static +g projection is R.T @ [0, 0, 1].
"""
import struct
import numpy as np


def parse_imu_packet(packet):
    if len(packet) != 20 or bytes(packet[:2]) != b'\x55\x61':
        return None
    values = struct.unpack('<9h', packet[2:20])
    return dict(acc=np.array(values[:3], dtype=float) / 32768 * 16,
                gyro=np.array(values[3:6], dtype=float) / 32768 * 2000,
                angles=np.array(values[6:9], dtype=float) / 32768 * 180)


def gravity_in_sensor(angles_deg):
    roll, pitch = np.deg2rad(angles_deg[:2])
    return np.array([-np.sin(pitch), np.sin(roll) * np.cos(pitch),
                     np.cos(roll) * np.cos(pitch)])


def linear_acceleration(packet, sensor_bias_g, gravity=9.81):
    return (packet['acc'] - gravity_in_sensor(packet['angles'])
            - np.asarray(sensor_bias_g)) * gravity


def calibration_quality(samples, residual_samples, gyro_samples, timestamps):
    """Shared two-second static calibration acceptance rules."""
    if len(samples) < 20 or timestamps[-1] - timestamps[0] < 1.9:
        return False, 'Waiting for 2 seconds of samples'
    raw = np.asarray(samples)
    residual = np.asarray(residual_samples)
    gyro = np.asarray(gyro_samples)
    half = len(residual) // 2
    drift = np.linalg.norm(residual[:half].mean(axis=0) - residual[half:].mean(axis=0))
    if not .8 <= np.linalg.norm(raw.mean(axis=0)) <= 1.2:
        return False, 'Keep IMU stationary (acceleration is not near 1 g)'
    if np.mean(np.linalg.norm(gyro, axis=1)) > 5 or np.linalg.norm(raw.std(axis=0)) > .03:
        return False, 'Motion detected: keep IMU still'
    if drift > .012 or np.linalg.norm(residual.std(axis=0)) > .02:
        return False, 'Attitude estimate is settling; keep IMU still'
    if np.linalg.norm(residual.mean(axis=0)) > .08:
        return False, 'Gravity/angle mismatch: waiting for attitude to settle'
    return True, 'Stable calibration samples'
