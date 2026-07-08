from datetime import datetime
from pathlib import Path

import qtawesome as qta
from PySide6 import QtCore, QtWidgets

from emg_interface.custom_components.path_edit import PathEdit
from emg_interface.docks.dock_base import BaseDock


class RecordDock(BaseDock):
    start_record = QtCore.Signal(Path)
    end_record = QtCore.Signal()

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Record Data")

        self._save_formats = {"CSV (*.csv)": ".csv"}
        self.icon_size = 24

        self.form_layout = QtWidgets.QFormLayout()
        self.dock_layout.addLayout(self.form_layout)

        row_layout = QtWidgets.QHBoxLayout()

        self.save_directory_edit = PathEdit(self)
        self.save_directory_edit.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.save_directory_edit.setToolTip("Save Dir")
        row_layout.addWidget(self.save_directory_edit)

        self.dir_button = QtWidgets.QPushButton(self)
        self.dir_button.setText("…")
        self.dir_button.setMaximumWidth(20)
        self.dir_button.setFocusPolicy(QtCore.Qt.NoFocus)
        self.dir_button.setToolTip("Get directory.")
        self.dir_button.clicked.connect(self.set_dir)
        row_layout.addWidget(self.dir_button)
        self.form_layout.addRow("Save Directory: ", row_layout)

        self.save_name_edit = QtWidgets.QLineEdit(self)
        self.form_layout.addRow("Save Name: ", self.save_name_edit)

        self.format_combobox = QtWidgets.QComboBox(self)
        self.format_combobox.addItems(self._save_formats.keys())
        self.form_layout.addRow("Save Format: ", self.format_combobox)

        row_layout = QtWidgets.QHBoxLayout()
        self.record_button = QtWidgets.QPushButton(self)
        self.record_icon = qta.icon("mdi.record-circle", color="red")
        self.recording_icon = qta.icon(
            "mdi.restore",
            color="red",
            animation=qta.Spin(self.record_button, interval=15, step=-15),
        )
        self.record_button.setIconSize(QtCore.QSize(self.icon_size, self.icon_size))
        self.record_button.setIcon(self.record_icon)
        self.record_button.setCheckable(True)
        self.record_button.setChecked(False)
        self.record_button.toggled.connect(self.record_button_toggled)
        self.record_button.setToolTip("Record")
        row_layout.addWidget(self.record_button)

        self.dock_layout.addLayout(row_layout)

    def record_button_toggled(self):
        if self.record_button.isChecked():
            save_dir = self.get_save_dir()
            save_name = self.get_save_name()
            save_format = self._save_formats[self.format_combobox.currentText()]
            save_path = save_dir / (save_name + save_format)
            if save_dir:
                if not save_path.exists():
                    self.start_record.emit(save_path)
                    self.record_button.setIcon(self.recording_icon)
                else:
                    reply = QtWidgets.QMessageBox.question(
                        self, "File exists!\n", "Do you want to overwrite the file?"
                    )
                    if reply == QtWidgets.QMessageBox.Yes:
                        self.start_record.emit(save_path)
                        self.record_button.setIcon(self.recording_icon)
                    else:
                        self.record_button.blockSignals(True)
                        self.record_button.setChecked(False)
                        self.record_button.blockSignals(False)
            else:
                self.record_button.blockSignals(True)
                self.record_button.setChecked(False)
                self.record_button.blockSignals(False)
        else:
            self.end_record.emit()
            self.record_button.setIcon(self.record_icon)

    def set_dir(self):
        if path := QtWidgets.QFileDialog.getExistingDirectory():
            path = Path(path).resolve()
            self.save_directory_edit.setText(str(path))

    def get_save_dir(self):
        save_dir = Path(self.save_directory_edit.text()).resolve()
        if save_dir.is_dir():
            self.save_directory_edit.setText(str(save_dir))
            return save_dir
        else:
            try:
                reply = QtWidgets.QMessageBox.question(
                    self,
                    "Make folder?",
                    "Folder does not exists.\nDo you want to make a folder?",
                )
                if reply != QtWidgets.QMessageBox.Yes:
                    return None
                save_dir.mkdir()
                self.save_directory_edit.setText(str(save_dir))
                return save_dir
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", str(e))
                self.set_dir()
                return None

    def get_save_name(self):
        if save_name := self.save_name_edit.text():
            return save_name
        else:
            return datetime.now().strftime("%Y-%m-%d_%H%M%S")

    def gui_save(self, settings):
        text = self.save_directory_edit.text()
        settings.setValue(f"{self.save_heading}/save_directory_edit", text)

    def gui_restore(self, settings):
        text = settings.value(f"{self.save_heading}/save_directory_edit")
        self.save_directory_edit.setText(text)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    widget = RecordDock()
    widget.show()

    app.exec()
