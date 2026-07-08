from PySide6 import QtWidgets

from emg_interface.docks.dock_base import BaseDock


class ResultAutocolor(BaseDock):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Result")

        self.bottun_layout = QtWidgets.QVBoxLayout()
        self.dock_layout.addLayout(self.bottun_layout)

        self.bottun_1 = QtWidgets.QPushButton("なし")
        self.bottun_1.setStyleSheet("font-size:44px")
        self.bottun_layout.addWidget(self.bottun_1)

    def result_bottun(self, data):
        if data == 0:
            self.bottun_1.setText("なし")
        elif data == 1:
            self.bottun_1.setText("握り")
        elif data == 2:
            self.bottun_1.setText("開き")
        elif data == 3:
            self.bottun_1.setText("尺屈")
        elif data == 4:
            self.bottun_1.setText("撓屈")


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    widget = ResultAutocolor()
    widget.show()

    app.exec()
