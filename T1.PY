import asyncio
from bleak import BleakClient


# ============================================================
# 配置
# ============================================================

IMU_ADDRESS = "D8:76:A8:5D:CC:CF"


# ============================================================
# 扫描设备的 Services / Characteristics
# ============================================================

async def main():

    print("=" * 70)
    print("WT9011DCL-BT50 BLE Characteristic Scanner")
    print("=" * 70)

    print()
    print(f"Device address: {IMU_ADDRESS}")
    print("正在连接...")
    print()

    try:

        async with BleakClient(IMU_ADDRESS) as client:

            print(
                f"Connected: {client.is_connected}"
            )

            print()
            print("=" * 70)
            print("BLE Services / Characteristics")
            print("=" * 70)

            # ------------------------------------------------
            # 获取所有 services
            # ------------------------------------------------

            services = client.services

            for service in services:

                print()
                print("-" * 70)

                print(
                    f"Service UUID:"
                )

                print(
                    f"  {service.uuid}"
                )

                if service.description:
                    print(
                        f"Description:"
                    )
                    print(
                        f"  {service.description}"
                    )

                # --------------------------------------------
                # Characteristics
                # --------------------------------------------

                for char in service.characteristics:

                    print()
                    print(
                        f"  Characteristic UUID:"
                    )

                    print(
                        f"    {char.uuid}"
                    )

                    print(
                        f"  Description:"
                    )

                    print(
                        f"    {char.description}"
                    )

                    print(
                        f"  Properties:"
                    )

                    for prop in char.properties:

                        print(
                            f"    - {prop}"
                        )

                    # ----------------------------------------
                    # 特别标记 WRITE
                    # ----------------------------------------

                    write_types = []

                    if "write" in char.properties:
                        write_types.append(
                            "WRITE"
                        )

                    if "write-without-response" in char.properties:
                        write_types.append(
                            "WRITE WITHOUT RESPONSE"
                        )

                    if write_types:

                        print()
                        print(
                            "  >>> ★ WRITE CAPABLE ★"
                        )

                        for wt in write_types:

                            print(
                                f"      {wt}"
                            )

            print()
            print("=" * 70)
            print("扫描完成")
            print("=" * 70)

            print()
            print(
                "请把整个终端输出复制给我。"
            )

    except Exception as e:

        print()
        print("=" * 70)
        print("连接失败")
        print("=" * 70)

        print(
            f"Error: {e}"
        )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    asyncio.run(main())