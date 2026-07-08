from PySide6 import QtWidgets, QtCore


class BaseDock(QtWidgets.QDockWidget):
    def __init__(self):
        super().__init__()

        self.setObjectName(self.__class__.__name__)
        self.save_heading = self.__class__.__name__

        self.dock_contents = QtWidgets.QFrame(parent=self)
        self.setWidget(self.dock_contents)
        self.setAllowedAreas(QtCore.Qt.AllDockWidgetAreas)
        self.setFeatures(
            self.DockWidgetFeature.DockWidgetFloatable
            | self.DockWidgetFeature.DockWidgetMovable
            | self.DockWidgetFeature.DockWidgetClosable
        )

        self.dock_layout = QtWidgets.QBoxLayout(
            QtWidgets.QBoxLayout.TopToBottom, self.dock_contents
        )


if __name__ == "__main__":
    app = QtWidgets.QApplication([])

    widget = BaseDock()
    widget.show()

    app.exec()
