import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets, QtGui


class RadarPlotWidget(QtWidgets.QWidget):
    def __init__(self, num_channels, parent=None):
        super().__init__(parent=parent)

        self.num_channels = num_channels
        self.main_layout = QtWidgets.QHBoxLayout(self)
        self.foreground_color = self.palette().color(self.foregroundRole())

        pg.setConfigOptions(
            background=None,
            foreground=self.foreground_color,
            antialias=True,
        )

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setAspectLocked()
        self.axis_theta = self._draw_axis(1, pg.mkPen(color=(0, 0, 0, 255), width=1))
        self.axis_theta = np.append(self.axis_theta, self.axis_theta[0])

        self.line = self.plot_widget.plot(
            pen=pg.mkPen(color=(44, 160, 44, 200), width=3)
        )

        self.recognition_result_text_item = pg.TextItem(
            "", anchor=(0.5, 0.5), color=(255, 255, 255, 255), fill=(44, 160, 44, 255)
        )
        self.recognition_result_text_item.setPos(0, 1.8)
        self.set_recognition_fontsize(24)
        self.plot_widget.addItem(self.recognition_result_text_item)

        self.main_layout.addWidget(self.plot_widget)

    def _draw_axis(self, r, pen):
        self.plot_widget.hideAxis("left")
        self.plot_widget.hideAxis("bottom")
        self.plot_widget.addLine(x=0, pen=pen)
        self.plot_widget.addLine(y=0, pen=pen)

        circle = QtWidgets.QGraphicsEllipseItem(-r, -r, r * 2, r * 2)
        circle.setPen(pen)
        self.plot_widget.addItem(circle)

        axis_theta = []
        for i in range(self.num_channels):
            theta = 2 * np.pi / self.num_channels * i
            axis_theta.append(theta)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            anchor = (0, 1) if theta < np.pi else (1, 0)
            text_item = pg.TextItem(
                f"Channel {i + 1}", anchor=anchor, color=pen.color()
            )
            text_item.setPos(x, y)
            self.plot_widget.addItem(text_item, ignoreBounds=True)
        return np.array(axis_theta)

    def set_data(self, data):
        r = np.append(data, data[0])

        x = r * np.cos(self.axis_theta)
        y = r * np.sin(self.axis_theta)
        self.line.setData(x, y)

    def set_recognition_result(self, text):
        self.recognition_result_text_item.setText(f" {text} ")

    def set_recognition_fontsize(self, fontsize):
        result_font = QtGui.QFont()
        result_font.setPointSize(fontsize)
        self.recognition_result_text_item.setFont(result_font)


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    widget = RadarPlotWidget(num_channels=4)
    test_data = np.array((0.5, 0.4, 0.2, 0.9))
    widget.set_data(test_data)

    widget.show()

    app.exec()
