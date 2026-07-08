import pyqtgraph as pg
from PySide6 import QtWidgets


class LivePlotWidget(QtWidgets.QWidget):
    def __init__(self, num_channels, parent=None):
        super().__init__(parent=parent)

        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.foreground_color = self.palette().color(self.foregroundRole())

        pg.setConfigOptions(
            background=None,
            foreground=self.foreground_color,
            antialias=True,
        )

        self.plot_layout = pg.GraphicsLayoutWidget()
        self.main_layout.addWidget(self.plot_layout)

        self.plots = []
        self.lines = []

        for i in range(num_channels):
            self._add_plot(f"Channel {i + 1}")

    def _add_plot(self, label):
        plot_widget = self.plot_layout.addPlot()
        self.plot_layout.nextRow()

        plot_widget.setLabel("left", label)
        plot_widget.getAxis("left").setWidth(80)

        plot_widget.setMenuEnabled(False)
        plot_widget.autoBtn.clicked.disconnect()
        plot_widget.autoBtn.clicked.connect(self._auto_range)
        plot_widget.setMouseEnabled(x=False, y=True)
        self.plots.append(plot_widget)

        line = plot_widget.plot(pen=self.foreground_color)
        self.lines.append(line)

    def _auto_range(self):
        plot_item = self.sender().parentItem()
        plot_item.enableAutoRange()

    def set_data(self, data):
        for i, l in enumerate(self.lines):
            l.setData(data[:, i])


if __name__ == "__main__":
    import numpy as np

    app = QtWidgets.QApplication([])
    widget = LivePlotWidget(num_channels=4)
    test_data = np.zeros((100, 4))
    widget.set_data(test_data)

    widget.show()

    app.exec()
