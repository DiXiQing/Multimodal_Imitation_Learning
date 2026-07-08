from scipy import signal

from emg_interface.defs import SAMPLING_RESOLUTION, VCC, SENSOR_GAIN


def adc_to_mV(y):
    emg_V = (y / SAMPLING_RESOLUTION - 0.5) * VCC / SENSOR_GAIN
    emg_mV = emg_V * 1000
    return emg_mV


def setup_envelope_filter(highcut, fs, order=3):  # カットオフ周波数は元々12Hz，dock_processingで変更可
    """Get coeffs for filter"""
    nyq = 0.5 * fs
    high = highcut / nyq
    return signal.butter(order, high, btype="lowpass", output="ba")


def apply_envelope(coefs, data):
    return signal.filtfilt(*coefs, data)


def setup_realtime_envelop_filter(high_cut, fs, order=4):
    b, a = setup_envelope_filter(highcut=high_cut, fs=fs, order=order)
    z = signal.lfilter_zi(b, a)
    return b, a, z


def realtime_filter(data, b, z, a=0):
    realtime_data, z = signal.lfilter(b, a, data, zi=z)
    return realtime_data, z
