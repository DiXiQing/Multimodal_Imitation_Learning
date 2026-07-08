from queue import Queue

import numpy as np
from PySide6 import QtCore

# from emg_interface.workers import SerialWorker

class RecognitionWorker(QtCore.QObject):
    recognised = QtCore.Signal(object) # 数値か文字

    def __init__(self):
        super().__init__()

        #self.llgmn = LLGMN()
        self.llgmn = None      # 元々こっち
        
        self.identity = {}
        self.input_queue = Queue()
        self.stop = False

    def run(self):
        while not self.stop:
            data = self.input_queue.get(block=True)
            data = data[-1, :]
            if self.llgmn is not None:
                output = self.llgmn.forward(np.array([data]))
                movement = str(output.argmax())  # 数値のみ
 
                # if data[0] == 1 and data[1] == 1:
                #     movement = str(0)

                if movement in self.identity:
                    movement = self.identity[movement]
                self.recognised.emit(movement)

    def update_model(self, llgmn):
        self.llgmn = llgmn

    def update_identity(self, identity):
        self.identity = identity

    def set_stop(self):
        self.stop = True


if __name__ == "__main__":
    worker = RecognitionWorker()
    worker.run()
