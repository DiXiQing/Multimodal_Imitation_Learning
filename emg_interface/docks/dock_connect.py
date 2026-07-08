import json

import qtawesome as qta
from PySide6 import QtWidgets, QtCore
from bitalino import find

from emg_interface.docks.dock_base import BaseDock


class ConnectDock(BaseDock):
    connect_clicked = QtCore.Signal(str)

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Connect to BITalino")
        self.devices = {}

        devices_layout = QtWidgets.QHBoxLayout()
        self.dock_layout.addLayout(devices_layout)

        self.devices_combobox = QtWidgets.QComboBox(self)
        self.devices_combobox.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        devices_layout.addWidget(self.devices_combobox)

        self.refresh_button = QtWidgets.QPushButton(self)
        self.refresh_button.setIcon(qta.icon("mdi.refresh"))
        self.refresh_button.setSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred
        )
        devices_layout.addWidget(self.refresh_button)

        self.output_label = QtWidgets.QLabel(self)
        self.output_label.setWordWrap(True)
        self.dock_layout.addWidget(self.output_label)

        self.connect_button = QtWidgets.QPushButton(self)
        self.connect_button.setIcon(qta.icon("mdi.link"))
        self.connect_button.setText("Connect")
        self.dock_layout.addWidget(self.connect_button)

        self.dock_layout.addStretch()

        self.devices_combobox.currentTextChanged.connect(self.show_mac_address)
        self.refresh_button.clicked.connect(self.refresh_devices)
        self.connect_button.clicked.connect(self.connect_button_clicked)

        # self.refresh_devices()
        # self.show_mac_address()

    def refresh_devices(self):
        self.devices.clear()
        self.devices_combobox.blockSignals(True)
        self.devices_combobox.clear()
        for mac_address, device_name in find():
            self.devices[device_name] = mac_address
        self.devices_combobox.addItems(self.devices.keys())
        self.devices_combobox.blockSignals(False)

        # set current to BITalino
        for k in self.devices.keys():
            if "bitalino" in k.lower():
                self.devices_combobox.setCurrentText(k)
                self.show_mac_address()

    def show_mac_address(self):
        device_name = self.devices_combobox.currentText()
        self.output_label.setText(f"MAC: {self.devices[device_name]}")

    def connect_button_clicked(self):
        device_name = self.devices_combobox.currentText()
        mac_address = self.devices[device_name]
        self.connect_clicked.emit(mac_address)

    def gui_save(self, settings):
        settings.setValue(f"{self.save_heading}/devices", json.dumps(self.devices))
        settings.setValue(
            f"{self.save_heading}/current_device", self.devices_combobox.currentText()
        )

    def gui_restore(self, settings):
        self.devices = json.loads(settings.value(f"{self.save_heading}/devices"))
        self.devices_combobox.addItems(self.devices.keys())
        self.devices_combobox.setCurrentText(
            settings.value(f"{self.save_heading}/current_device")
        )


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    widget = ConnectDock()
    widget.show()

    app.exec()
