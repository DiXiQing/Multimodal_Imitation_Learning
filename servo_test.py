"""
servo_move.py
测试 STS3215 转动
让舵机在两个位置之间来回转动
"""

from scservo_sdk import *
import time

PORT = "COM8"
BAUD = 1000000
SERVO_ID = 1

# STS3215 寄存器地址
ADDR_TORQUE_ENABLE   = 40
ADDR_GOAL_POSITION   = 42
ADDR_PRESENT_POSITION = 56


def main():
    portHandler = PortHandler(PORT)
    packetHandler = PacketHandler(protocol_end=0)

    if not portHandler.openPort():
        print(f"[Error] 无法打开端口 {PORT}")
        return
    portHandler.setBaudRate(BAUD)
    print(f"[OK] 连接成功\n")

    # 使能扭矩
    packetHandler.write1ByteTxRx(portHandler, SERVO_ID, ADDR_TORQUE_ENABLE, 1)
    print("扭矩已使能，开始转动测试\n")

    # 在 1000 和 3000 之间来回转
    positions = [1000, 3000, 2048]
    for pos in positions:
        print(f"移动到位置 {pos} ({pos/4095*360:.0f} deg)")
        # 写目标位置（2字节）
        packetHandler.write2ByteTxRx(portHandler, SERVO_ID, ADDR_GOAL_POSITION, pos)
        time.sleep(1.5)

        # 读实际位置
        actual, comm, err = packetHandler.read2ByteTxRx(
            portHandler, SERVO_ID, ADDR_PRESENT_POSITION)
        print(f"  实际到达: {actual}\n")

    # 关闭扭矩
    packetHandler.write1ByteTxRx(portHandler, SERVO_ID, ADDR_TORQUE_ENABLE, 0)
    portHandler.closePort()
    print("测试完成")


if __name__ == "__main__":
    main()