import numpy as np
from PySide6 import QtWidgets, QtCore


class CalibrationGroupBox(QtWidgets.QGroupBox):
    start_calibration = QtCore.Signal()
    finish_calibration = QtCore.Signal()
    cal_value_changed = QtCore.Signal(list)

    def __init__(self, num_channels=4, duration=5, parent=None):
        super().__init__(parent=parent)

        layout = QtWidgets.QVBoxLayout(self)

        self.num_channels = num_channels
        self.duration = duration
        self.cal_values = np.zeros(self.num_channels)

        self.button_time_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(self.button_time_layout)

        self.button = QtWidgets.QPushButton("測定開始", self)
        self.button.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        self.button_time_layout.addWidget(self.button)

        self.countdown_label = QtWidgets.QLabel(self)
        self.countdown_label.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum
        )
        self.button_time_layout.addWidget(self.countdown_label)

        self.cal_values_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(self.cal_values_layout)

        self.spinboxes = []
        for i in range(self.num_channels):
            spinbox = QtWidgets.QDoubleSpinBox(self)
            spinbox.setDecimals(3)
            spinbox.setSingleStep(1e-3)
            spinbox.setValue(0)
            spinbox.setRange(-10, 10)
            spinbox.valueChanged.connect(self.spinbox_value_changed)
            self.cal_values_layout.addWidget(spinbox)
            self.spinboxes.append(spinbox)

        self.cal_timer = QtCore.QTimer()
        self.cal_timer.setSingleShot(True)
        self.cal_timer.timeout.connect(self.calibration_time_end)

        self.countdown_label_timer = QtCore.QTimer()
        self.countdown_label_timer.timeout.connect(self.update_countdown_label)

        self.button.clicked.connect(self.button_clicked)

    def button_clicked(self):
        self.button.setText("計測中")
        self.setDisabled(True)
        self.start_calibration.emit()

        self.cal_timer.start(self.duration * 1000)
        self.countdown_label_timer.start(30)

    def calibration_time_end(self):
        self.button.setText("測定開始")
        self.setDisabled(False)
        self.countdown_label_timer.stop()
        self.countdown_label.setText("")
        self.finish_calibration.emit()

    def spinbox_value_changed(self):
        self.cal_values = [sb.value() for sb in self.spinboxes]
        self.cal_value_changed.emit(self.cal_values)

    def set_cal_values(self, values):
        for sb, val in zip(self.spinboxes, values):
            sb.blockSignals(True)
            sb.setValue(val)
            sb.blockSignals(False)
        self.spinbox_value_changed()

    def update_countdown_label(self):
        self.countdown_label.setText(
            f"残り: {self.cal_timer.remainingTime() / 1000:.2f} s"
        )

    def set_button_tooltip(self, text):
        self.button.setToolTip(text)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    widget = CalibrationGroupBox()
    widget.show()
    app.exec()
