#!/usr/bin/env python3
"""测试松灵 Piper 与 CAN 通信是否正常。

只读取关节角、不动机械臂、不使能，非常安全。
用法：conda activate lerobot && python 03_test_piper.py
"""
import sys
import time

CAN_INTERFACE = "can0"  # 与 config.env 保持一致


def main() -> None:
    try:
        from piper_sdk import C_PiperInterface  # type: ignore
    except ImportError:
        print("[✗] 未安装 piper_sdk，请先运行: bash 02_install_env.sh")
        sys.exit(1)

    # 兼容不同版本 SDK：新版可传 CAN 接口名，旧版无参构造
    try:
        piper = C_PiperInterface(CAN_INTERFACE)
    except TypeError:
        piper = C_PiperInterface()

    piper.ConnectPort()
    print(f"[✓] 已连接 {CAN_INTERFACE}，开始读取关节角（5 秒）...")
    print("    如果下面能打印出关节角度，说明 CAN 通信正常。Ctrl+C 退出。")

    start = time.time()
    while time.time() - start < 5:
        try:
            joint = piper.GetArmJointMsgs()
            print(f"关节角: {joint}")
        except Exception as e:  # noqa: BLE001
            print(f"[!] 读取失败: {e}")
        time.sleep(0.2)

    print("测试结束。能持续打印出数值即代表硬件链路 OK。")


if __name__ == "__main__":
    main()
