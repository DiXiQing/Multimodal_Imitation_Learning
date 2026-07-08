import serial
from PySide6 import QtCore, QtWidgets
from serial.tools import list_ports

from emg_interface.docks.dock_base import BaseDock
from emg_interface.workers import ArduinoWorker


class ArduinoDock(BaseDock):
    find_device_failed = QtCore.Signal(str)
    device_found = QtCore.Signal(object)

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Arduino")
        self.device_name = "Arduino"
        self.baudrate = 9600
        self.port = None

        self.connect_button = QtWidgets.QPushButton(self)
        self.connect_button.setText("Connect")
        self.connect_button.clicked.connect(self.connect_button_clicked)
        self.dock_layout.addWidget(self.connect_button)

        self.scrollable_area = QtWidgets.QScrollArea(self)
        self.scrollable_area.setFrameStyle(QtWidgets.QFrame.Shadow.Plain)
        self.dock_layout.addWidget(self.scrollable_area)

        self.output_label = QtWidgets.QLabel(self)
        self.scrollable_area.setWidget(self.output_label)

        # thread for streaming arduino data
        self.arduino_thread = QtCore.QThread()
        self.arduino_worker = None

    # TODO:连接?
    def connect_button_clicked(self):
        port_found = find_port(self.device_name)
        if port_found is not None:
            self.port = port_found
            ser = serial.Serial(port=self.port, baudrate=self.baudrate, timeout=5)
            self.device_found.emit(ser)
        else:
            self.find_device_failed.emit("Arduino not found!")
            print("ハンドを接続してください")
            self.ser = None

    def display_output(self, msg):
        t = self.output_label.text()
        self.output_label.setText(f"{t}{msg}\n")
        self.output_label.adjustSize()
        self.scrollable_area.verticalScrollBar().setSliderPosition(
            self.scrollable_area.verticalScrollBar().maximum()
        )
        

def find_port(device_name):
    for port, desc, _ in list_ports.comports():
        if device_name in desc:
            break
    else:
        port = None
    return port


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    widget = ArduinoDock()
    widget.show()

    app.exec()
