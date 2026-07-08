from PySide6 import QtWidgets, QtCore

from emg_interface.docks.dock_base import BaseDock
from emg_interface.groupboxes import CalibrationGroupBox


class ProcessingDock(BaseDock):
    processing_updated = QtCore.Signal(dict)
    start_zero_offset_calibration = QtCore.Signal()
    finish_zero_offset_calibration = QtCore.Signal()
    zero_offset_changed = QtCore.Signal(list)
    start_max_value_calibration = QtCore.Signal()
    finish_max_value_calibration = QtCore.Signal()
    max_value_changed = QtCore.Signal(list)

    def __init__(self, num_channels=4):
        super().__init__()

        self.num_channels = num_channels

        self.setWindowTitle("Processing")

        self.rectify_checkbox = QtWidgets.QCheckBox("Rectify", self)
        self.dock_layout.addWidget(self.rectify_checkbox)

        self.envelope_checkbox = QtWidgets.QCheckBox("Envelope", self)
        self.dock_layout.addWidget(self.envelope_checkbox)

        self.max_min_checkbox = QtWidgets.QCheckBox("Max Min Scaling", self)
        self.dock_layout.addWidget(self.max_min_checkbox)

        self.normalise_checkbox = QtWidgets.QCheckBox("Channel Normalise", self)
        self.dock_layout.addWidget(self.normalise_checkbox)

        self.processes = {
            "rectify": False,
            "envelop": False,
            "max_min_scaling": False,
            "channel_normalise": False,
        }

        self.dock_layout.addStretch()

        self.zero_offset_groupbox = CalibrationGroupBox(  #normal 3秒間
            num_channels, duration=3, parent=self
        )
        self.zero_offset_groupbox.setTitle("Zero Offset Calibration")
        self.zero_offset_groupbox.set_button_tooltip("!! TO DO")
        self.zero_offset_groupbox.start_calibration.connect(
            self.zero_offset_calibration_started
        )
        self.zero_offset_groupbox.finish_calibration.connect(
            self.zero_offset_calibration_finished
        )
        self.zero_offset_groupbox.cal_value_changed.connect(
            self.zero_offset_changed.emit
        )
        self.dock_layout.addWidget(self.zero_offset_groupbox)

        self.max_value_groupbox = CalibrationGroupBox(  #maxmin 7秒間
            num_channels, duration=7, parent=self
        )
        self.max_value_groupbox.setTitle("Max Value Calibration")
        self.max_value_groupbox.set_button_tooltip("!! TO DO")
        self.max_value_groupbox.set_cal_values([1] * num_channels)
        self.max_value_groupbox.start_calibration.connect(
            self.max_value_calibration_started
        )
        self.max_value_groupbox.finish_calibration.connect(
            self.max_value_calibration_finished
        )
        self.max_value_groupbox.cal_value_changed.connect(self.max_value_changed.emit)
        self.dock_layout.addWidget(self.max_value_groupbox)

        self.rectify_checkbox.toggled.connect(self.checkbox_selection_changed)
        self.envelope_checkbox.toggled.connect(self.checkbox_selection_changed)
        self.max_min_checkbox.toggled.connect(self.checkbox_selection_changed)
        self.normalise_checkbox.toggled.connect(self.checkbox_selection_changed)

    def checkbox_selection_changed(self):  # チェックボックス連鎖
        order = [
            self.rectify_checkbox,
            self.envelope_checkbox,
            self.max_min_checkbox,
            self.normalise_checkbox,
        ]  # must be processed in this order, without skipping steps
        i = order.index(self.sender())

        for checkbox in order[:i]:
            checkbox.blockSignals(True)
            checkbox.setChecked(True)
            checkbox.blockSignals(False)
        for checkbox in order[i + 1:]:
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)

        self.processes = {
            "rectify": self.rectify_checkbox.isChecked(),
            "envelop": self.envelope_checkbox.isChecked(),
            "max_min_scaling": self.max_min_checkbox.isChecked(),
            "channel_normalise": self.normalise_checkbox.isChecked(),
        }
        self.processing_updated.emit(self.processes)

    def zero_offset_calibration_started(self):
        self.max_value_groupbox.setDisabled(True)
        self.start_zero_offset_calibration.emit()

    def zero_offset_calibration_finished(self):
        self.max_value_groupbox.setDisabled(False)
        self.finish_zero_offset_calibration.emit()

    def calc_zero_offsets(self, collected_data):
        zero_offsets = collected_data.mean(axis=0)
        self.zero_offset_groupbox.set_cal_values(zero_offsets)

    def max_value_calibration_started(self):
        self.zero_offset_groupbox.setDisabled(True)
        self.start_max_value_calibration.emit()

    def max_value_calibration_finished(self):
        self.zero_offset_groupbox.setDisabled(False)
        self.finish_max_value_calibration.emit()

    def calc_max_values(self, collected_data):
        max_values = collected_data.max(axis=0)
        self.max_value_groupbox.set_cal_values(max_values)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    widget = ProcessingDock(4)
    widget.show()

    app.exec()
