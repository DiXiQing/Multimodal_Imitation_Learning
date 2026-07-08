from queue import Queue
import serial

from PySide6 import QtCore


class ArduinoWorker(QtCore.QObject):

    def __init__(self, ser):
        super().__init__()

        self.ser = ser
        # self.ser = serial.Serial('COM10',9600,timeout=None)

        self.result_queue = Queue()  # 判別結果
        self.zizyou_queue = Queue()  # ２乗で閾値
        self.powersize_queue = Queue()  # handのpowerの強さ
        self.stop = False

    def test(self):
        self.ser.write("1".encode())

    def run(self):
        while not self.stop:
            recog_result = self.result_queue.get(block=True)
            print("recog_result")
            print(recog_result)

            if recog_result == "Close":
                self.ser.write("1".encode())
            elif recog_result == "Open":
                self.ser.write("2".encode())
            elif recog_result == "Left":
                self.ser.write("3".encode())
            elif recog_result == "Right":
                self.ser.write("4".encode())
            # elif recog_result == "Right":
            #     self.ser.write("5".encode())


    def stop_run(self):
        self.stop = True

if __name__ == "__main__":
    ser = serial.Serial(port="COM4", baudrate=9600, timeout=5)

    worker = ArduinoWorker(ser)
    worker.test()
