"""
通过 BITalino 设备读取 EMG（肌电）信号、实时处理并可视化、保存或用于其他分析用途。
"""

import csv
import numpy as np
from PySide6 import QtCore
from bitalino import BITalino

from emg_interface.funcs import (
    adc_to_mV,
    setup_realtime_envelop_filter,
    realtime_filter,
)


class StreamWorker(QtCore.QObject):
    finished = QtCore.Signal()
    stream_error = QtCore.Signal(str)
    collection_complete = QtCore.Signal(object)
    downsampled_data = QtCore.Signal(object)

    def __init__(
            self,
            mac_address,
            window_length_seconds=5,
            sampling_rate=1000,
            num_channels=4,
            chunk_size=10,
    ):
        super().__init__()

        self.mac_address = mac_address
        self.sampling_rate = sampling_rate
        self.window_length_seconds = window_length_seconds
        self.processing = {
            "rectify": False,
            "envelop": False,
            "max_min_scaling": False,
            "channel_normalise": False,
        }
        self.num_channels = num_channels
        self.chunk_size = chunk_size

        self.window_length_samples = self.window_length_seconds * self.sampling_rate

        self.raw_buffer = np.zeros((self.window_length_samples, self.num_channels))
        self.processed_buffer = np.zeros(
            (self.window_length_samples, self.num_channels)
        )

        self.b_i, self.a_i, self.z_i = self.set_envelop_cutoff_freq(1)  # カットオフ周波数1Hz

        self.stop = False
        self.zero_offset = np.zeros(self.num_channels)
        self.max_scaling = np.ones(self.num_channels)
        self.channel_scaling = np.ones(self.num_channels)

        self.collect_flag = False
        self.collected = []

        self.downsampling_prescaler = 1  # 10 Hz (100 Hz [Batch update rate] / 10)
        self.downsampling_counter = 0

        self.record_file = None
        self.record_writer = None

        self.acqChannels = [1, 2, 3, 4]

        self.mutex = QtCore.QMutex()

        self.power_size = 0

    def stream(self):
        device = None

        print_count = 0

        try:
            device = BITalino(self.mac_address)
            # device.start(self.sampling_rate, list(range(self.num_channels)))
            device.start(self.sampling_rate, self.acqChannels)
        except Exception as e:
            self.stream_error.emit(str(e))

            if device is not None:
                device.close()
                device = None
            self.stop = True

        while not self.stop:
            try:
                raw_data = device.read(self.chunk_size)[:, 5:]
                raw_data = adc_to_mV(raw_data)
                # print("Raw data:", raw_data)  # Debugging: print raw data
            except Exception as e:
                self.stream_error.emit(str(e))
                self.stop = True
                break

            processed_data = raw_data.copy()
            if self.processing["rectify"]:
                processed_data = np.abs(processed_data)  # 整流
                # print("Rectified data:", processed_data)  # Debugging: print rectified data
            if self.processing["envelop"]:  # ローパスフィルタ
                processed_data = self.envelop_filter(processed_data)
                # print("Enveloped data:", processed_data)  # Debugging: print enveloped data

            if self.processing["max_min_scaling"]:  # minを0、maxを1とする
                processed_data = processed_data - self.zero_offset  # 無力からの差分処理
                processed_data = processed_data / self.max_scaling
                zero = processed_data[-1, :]
                zero = sum(zero)

                # zizyou = sum(map(lambda x: x**2, processed_data[-1,:]))
                # if zizyou > 0.5:  # 二乗和（しきい値｛ver1｝ 3move=0.03，5move=0.012）
                #     processed_data = np.ones(40).reshape(10, 4)

            if self.processing["channel_normalise"]:  # 正規化
                if zero < 0.001:  # 閾値、動作とみなすかどうか　0.29
                    processed_data = np.ones((10, self.num_channels), int)
                    self.power_size = 0
                elif zero > 0.8:
                    self.power_size = 3
                elif zero > 0.6:
                    self.power_size = 2
                elif zero > 0.3:
                    self.power_size = 1
                processed_data = processed_data / processed_data.sum(axis=1)[:, None]
                # print("Channel normalized data:", processed_data)  # Debugging: print channel normalized data

            # roll data
            self.mutex.lock()
            self.raw_buffer[: -self.chunk_size, :] = self.raw_buffer[
                                                     self.chunk_size:, :
                                                     ]
            self.raw_buffer[-self.chunk_size:, :] = raw_data
            self.mutex.unlock()

            # processed_data
            self.mutex.lock()
            self.processed_buffer[: -self.chunk_size, :] = self.processed_buffer[
                                                           self.chunk_size:, :
                                                           ]
            self.processed_buffer[-self.chunk_size:, :] = processed_data
            self.mutex.unlock()

            if self.collect_flag:  # 押されたとき配列保存
                self.collected.append(processed_data)

            if self.downsampling_counter >= self.downsampling_prescaler:
                self.downsampled_data.emit(processed_data)
                self.downsampling_counter = 0
            else:
                self.downsampling_counter += 1

            if self.record_writer is not None:
                self.record_data(raw_data, processed_data)

        if device is not None:
            device.stop()
            device.close()
        self.downsampling_counter = 0
        self.stop = False
        self.finished.emit()

    def update_processing(self, new_processing):
        self.processing.update(new_processing)

    def set_envelop_cutoff_freq(self, freq):
        b_i, a_i, z_i = setup_realtime_envelop_filter(freq, fs=self.sampling_rate)
        return b_i, a_i, [z_i] * self.num_channels

    def envelop_filter(self, data):
        filtered_data = np.zeros((self.chunk_size, self.num_channels))
        for j in range(self.num_channels):
            filtered_data[:, j], self.z_i[j] = realtime_filter(
                data[:, j], self.b_i, self.z_i[j], self.a_i
            )  # フィルター処理
        return filtered_data

    def set_stop(self):
        self.stop = True

    def start_collect_data(self):
        self.collect_flag = True

    def set_zero_offsets(self, offsets):
        self.zero_offset = offsets

    def set_max_scaling(self, max_scaling):
        self.max_scaling = max_scaling

    def finish_collect_data(self):
        self.collect_flag = False
        self.collection_complete.emit(np.concatenate(self.collected, axis=0))
        self.collected.clear()

    def start_record(self, path):
        self.record_file = open(path, mode="w", encoding="utf-8", newline="")
        self.record_writer = csv.writer(self.record_file, delimiter=",")
        self.record_writer.writerow(
            [f"raw_ch{i + 1}" for i in range(self.num_channels)]
            + [f"processed_ch{i + 1}" for i in range(self.num_channels)]
        )

    def record_data(self, raw_data, processed_data):
        self.record_writer.writerows(np.concatenate([raw_data, processed_data], axis=1))

    def finish_record(self):
        self.record_file.close()
        self.record_writer = None
        self.record_file = None


if __name__ == "__main__":
    from time import sleep

    worker = StreamWorker("00:21:06:BE:18:0B")
    worker.stream()
    # print(worker.raw_buffer)

    sleep(1)
    worker.set_stop()
