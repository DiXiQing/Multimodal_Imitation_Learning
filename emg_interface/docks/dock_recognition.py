import json
from pathlib import Path

import qtawesome as qta
from PySide6 import QtWidgets, QtCore

from emg_interface.custom_components.path_edit import PathEdit
from emg_interface.docks.dock_base import BaseDock
from llgmn import LLGMN


class RecognitionDock(BaseDock):
    model_updated = QtCore.Signal(object)
    identity_updated = QtCore.Signal(object)

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Recognition")

        self.llgmn = LLGMN()
        self.identity = {}

        self.form_layout = QtWidgets.QFormLayout()
        self.dock_layout.addLayout(self.form_layout)

        row = QtWidgets.QHBoxLayout()
        self.form_layout.addRow("Weights: ", row)

        self.weight_ok = False

        self.weight_file_path = PathEdit("file", self)
        self.weight_file_path.acceptDrops()
        self.weight_file_path.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.weight_file_path.textChanged.connect(self.update_weights)
        self.weight_file_path.setToolTip("LLGMN weights file path.")
        row.addWidget(self.weight_file_path)

        dir_button = QtWidgets.QPushButton(self)
        dir_button.setText("…")
        dir_button.setMaximumWidth(20)
        dir_button.setFocusPolicy(QtCore.Qt.NoFocus)
        dir_button.clicked.connect(self.set_weights_file)
        dir_button.setToolTip("Get weights file.")
        row.addWidget(dir_button)

        self.icon_size = 18
        self.status_label = QtWidgets.QLabel(self)
        self.cross_icon = qta.icon("mdi.close-circle", color="red")
        self.tick_icon = qta.icon("mdi.check-circle", color="green")
        self.status_label.setPixmap(self.cross_icon.pixmap(self.icon_size))
        self.status_label.setToolTip("Weights load unsuccessful.")
        row.addWidget(self.status_label)

        row = QtWidgets.QHBoxLayout()
        self.form_layout.addRow("Identity: ", row)

        self.identity_file_path = PathEdit("file", self)
        self.identity_file_path.acceptDrops()
        self.identity_file_path.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.identity_file_path.textChanged.connect(self.update_identity)
        self.identity_file_path.setToolTip("output identity file path.")
        row.addWidget(self.identity_file_path)

        dir_button = QtWidgets.QPushButton(self)
        dir_button.setText("…")
        dir_button.setMaximumWidth(20)
        dir_button.setFocusPolicy(QtCore.Qt.NoFocus)
        dir_button.clicked.connect(self.set_identity_file)
        dir_button.setToolTip("Get identity file.")
        row.addWidget(dir_button)

        self.dock_layout.addStretch()

    def set_weights_file(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName()
        if file_name:
            self.weight_file_path.setText(file_name)

    def set_identity_file(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName()
        if file_name:
            self.identity_file_path.setText(file_name)

    def set_weights_ok(self):
        self.weight_ok = True
        self.status_label.setPixmap(self.tick_icon.pixmap(self.icon_size))
        self.status_label.setToolTip("Weights load successful.")

    def set_weights_bad(self):
        self.weight_ok = False
        self.status_label.setPixmap(self.cross_icon.pixmap(self.icon_size))
        self.status_label.setToolTip("Weights load unsuccessful.")

    def update_weights(self):
        try:
            file_name = self.weight_file_path.text()
            self.llgmn.load_weight(Path(file_name).resolve())
            self.set_weights_ok()

        except Exception as e:
            self.set_weights_bad()
        self.model_updated.emit(self.llgmn)

    def update_identity(self):
        try:
            file_name = self.identity_file_path.text()
            with open(Path(file_name).resolve(), encoding="utf-8") as f:
                self.identity = json.load(f)
        except Exception as e:
            print(e)
        self.identity_updated.emit(self.identity)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    widget = RecognitionDock()
    widget.show()

    app.exec()
