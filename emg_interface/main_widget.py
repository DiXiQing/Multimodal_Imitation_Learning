from PySide6 import QtWidgets, QtCore

from emg_interface.defs import SAMPLING_RATE, NUM_CHANNELS, CHUNK_SIZE, settings_file
from emg_interface.display_widget import LivePlotWidget, RadarPlotWidget
from emg_interface.docks import (
    ConnectDock,
    ProcessingDock,
    ArduinoDock,
    RecordDock,
    RecognitionDock,
)
from emg_interface.workers import StreamWorker, RecognitionWorker ,ArduinoWorker


class MainWidget(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("BITalino EMG")
        self.resize(800, 600)

        self.main_widget = QtWidgets.QWidget(self)
        self.main_layout = QtWidgets.QHBoxLayout(self.main_widget)
        self.setCentralWidget(self.main_widget)

        self.main_splitter = QtWidgets.QSplitter(self)
        self.main_layout.addWidget(self.main_splitter)

        self.live_plot_widget = LivePlotWidget(NUM_CHANNELS)
        self.main_splitter.addWidget(self.live_plot_widget)

        self.radar_plot_widget = RadarPlotWidget(NUM_CHANNELS)
        self.radar_plot_widget.set_recognition_result("なし")
        self.radar_plot_widget.set_recognition_fontsize(24)
        self.main_splitter.addWidget(self.radar_plot_widget)

        self.connect_dock = ConnectDock()
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.connect_dock)
        self.connect_dock.connect_clicked.connect(self.connect_bitalino)

        self.processing_dock = ProcessingDock(num_channels=NUM_CHANNELS)
        self.processing_dock.processing_updated.connect(self.update_processing)
        self.processing_dock.start_zero_offset_calibration.connect(
            self.start_zero_offset_calibration
        )
        self.processing_dock.finish_zero_offset_calibration.connect(
            self.finish_zero_offset_calibration
        )
        self.processing_dock.zero_offset_changed.connect(self.zero_offset_changed)
        self.processing_dock.start_max_value_calibration.connect(
            self.start_max_value_calibration
        )
        self.processing_dock.finish_max_value_calibration.connect(
            self.finish_max_value_calibration
        )
        self.processing_dock.max_value_changed.connect(self.max_values_changed)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.processing_dock)

        self.record_dock = RecordDock()
        self.record_dock.start_record.connect(self.start_record)
        self.record_dock.end_record.connect(self.end_record)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.record_dock)

        self.recognition_dock = RecognitionDock()
        self.recognition_dock.model_updated.connect(self.update_model)
        self.recognition_dock.identity_updated.connect(self.update_identity)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.recognition_dock)

        self.arduino_dock = ArduinoDock()
        self.arduino_dock.device_found.connect(self.connect_arduino)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.arduino_dock)
        self.lines = []

        # thread for streaming input
        self.stream_thread = QtCore.QThread()
        self.stream_worker = None

        # thread for recognition
        self.recognition_thread = QtCore.QThread()
        self.recognition_worker = RecognitionWorker()
        self.recognition_worker.moveToThread(self.recognition_thread)
        self.recognition_worker.recognised.connect(self.movement_recognised)
        # self.recognition_worker.serialed.connect(self.movement_serialed)
        # self.recognition_worker.serialed.connect(self.movement_zizyou)
        # self.recognition_worker.serialed.connect(self.movement_powersize)
        self.recognition_thread.started.connect(self.recognition_worker.run)
        self.recognition_thread.start()

        self.arduino_thread = QtCore.QThread()
        self.arduino_worker = None

        # periodically update display widgets
        self.plot_update_timer = QtCore.QTimer()
        self.plot_update_timer.timeout.connect(self.update_plots)
        self.plot_update_timer.start(30)

        # load settings from previous session
        self.settings_file = settings_file()
        if self.settings_file.is_file():
            settings = QtCore.QSettings(
                str(self.settings_file), QtCore.QSettings.IniFormat
            )
            self.gui_restore(settings)

    def connect_bitalino(self, mac_address):
        if self.stream_worker is not None:
            self.stream_worker.set_stop()

        self.stream_worker = StreamWorker(
            mac_address,
            window_length_seconds=5,
            sampling_rate=SAMPLING_RATE,
            num_channels=NUM_CHANNELS,
            chunk_size=CHUNK_SIZE,
        )
        self.stream_worker.moveToThread(self.stream_thread)
        self.stream_worker.finished.connect(self.stream_finished)
        self.stream_worker.stream_error.connect(self.handle_stream_error)
        self.stream_worker.downsampled_data.connect(self.gesture_recognition)
        self.stream_thread.started.connect(self.stream_worker.stream)
        self.stream_thread.start()

    def connect_arduino(self, ser):
        if self.arduino_worker is not None:
            self.arduino_worker.stop_run()

        print("trigger")
        
        if ser.is_open:
            print("Arduino connected")
        else:
            print("Arduino not connected")

        # arduinoのシリアル通信用thread
        self.arduino_worker = ArduinoWorker(ser)
        self.arduino_worker.moveToThread(self.arduino_thread)
        self.arduino_thread.started.connect(self.arduino_worker.run)
        self.arduino_thread.start()

    def update_processing(self, new_processing_flags):
        if self.stream_worker is not None:
            self.stream_worker.update_processing(new_processing_flags)

    def start_zero_offset_calibration(self):
        if self.stream_worker is not None:
            self.stream_worker.start_collect_data()

    def finish_zero_offset_calibration(self):
        if self.stream_worker is not None:
            self.stream_worker.collection_complete.connect(
                self.zero_offset_collection_complete
            )
            self.stream_worker.finish_collect_data()

    def zero_offset_collection_complete(self, collected_data):
        self.stream_worker.collection_complete.disconnect(
            self.zero_offset_collection_complete
        )
        self.processing_dock.calc_zero_offsets(collected_data)

    def zero_offset_changed(self, offsets):
        if self.stream_worker is not None:
            self.stream_worker.set_zero_offsets(offsets)

    def start_max_value_calibration(self):
        if self.stream_worker is not None:
            self.stream_worker.start_collect_data()

    def finish_max_value_calibration(self):
        if self.stream_worker is not None:
            self.stream_worker.collection_complete.connect(
                self.max_value_collection_complete
            )
            self.stream_worker.finish_collect_data()

    def max_value_collection_complete(self, collected_data):
        self.stream_worker.collection_complete.disconnect(
            self.max_value_collection_complete
        )
        self.processing_dock.calc_max_values(collected_data)

    def max_values_changed(self, max_values):
        if self.stream_worker is not None:
            self.stream_worker.set_max_scaling(max_values)

    def stream_finished(self):
        self.stream_thread.exit()
        self.stream_worker = None

    def handle_stream_error(self, msg):
        self.error_dialog(msg)

    def update_plots(self):
        if self.stream_worker is not None:
            self.live_plot_widget.set_data(self.stream_worker.processed_buffer)
            self.radar_plot_widget.set_data(self.stream_worker.processed_buffer[-1, :])

    def update_model(self, llgmn):
        self.recognition_worker.update_model(llgmn)

    def update_identity(self, identity):
        self.recognition_worker.update_identity(identity)

    def gesture_recognition(self, data):
        self.recognition_worker.input_queue.put(data)

    def movement_recognised(self, result):
        self.radar_plot_widget.set_recognition_result(result)
        # print("result")
        # print(result)
        if self.arduino_worker is not None:
            self.arduino_worker.result_queue.put(result)  # result:recognition_workerの数値のみの判別結果
    
    def movement_zizyou(self):
        self.arduino_worker.zizyou_queue.put(self.stream_worker.processed_buffer[-1, :])

    # def movement_powersize(self):
    #     self.arduino_worker.powersize_queue.put(self.stream_worker.power_size)

    def start_record(self, path):
        if self.stream_worker is not None:
            self.stream_worker.start_record(path)

    def end_record(self):
        if self.stream_worker is not None:
            self.stream_worker.finish_record()

    def gui_save(self, settings):
        self.connect_dock.gui_save(settings)
        settings.setValue("Window/geometry", self.saveGeometry())
        settings.setValue("Window/state", self.saveState())
        settings.setValue("Window/splitter", self.main_splitter.saveState())

    def gui_restore(self, settings):
        try:
            if geometry := settings.value("Window/geometry"):
                self.restoreGeometry(geometry)
            if state := settings.value("Window/state"):
                self.restoreState(state)
            if state := settings.value("Window/splitter"):
                self.main_splitter.restoreState(state)
            self.connect_dock.gui_restore(settings)

        except Exception as e:
            self.error_dialog(f"{self.settings_file} is corrupted!\n{str(e)}")

    def closeEvent(self, event):
        if self.stream_worker is not None:
            self.stream_worker.set_stop()
        settings = QtCore.QSettings(str(self.settings_file), QtCore.QSettings.IniFormat)
        self.gui_save(settings)
        event.accept()

    def error_dialog(self, error):
        QtWidgets.QMessageBox.critical(self, "Error", error)


def main():
    app = QtWidgets.QApplication([])
    win = MainWidget()
    win.show()

    app.exec()


if __name__ == "__main__":
    main()
