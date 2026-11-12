#!/usr/bin/env python3
"""手柄自检：线程读事件 + 主线程计时打印，验证 real_collect 的输入链路。

用法：python test_gamepad.py [--seconds 12]
推摇杆、按 A/B 键，看终端有没有反应。
"""
import argparse
import sys
import threading
import time

from sim_collect_piper import stick_to_delta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--stick_scale", type=float, default=0.002)
    args = ap.parse_args()

    try:
        import inputs
    except ImportError:
        print("❌ 未安装 inputs 库：pip install inputs")
        sys.exit(1)

    pads = inputs.devices.gamepads
    print(f"=== 检测到 {len(pads)} 个手柄 ===")
    for p in pads:
        print(f"  {p}")
    if not pads:
        print("❌ inputs 没找到手柄。检查 /dev/input 权限与 xpad 驱动")
        sys.exit(1)

    state = {"stick": {"ABS_X": 0, "ABS_Y": 0, "ABS_RX": 0, "ABS_RY": 0}, "keys": [], "events": 0}
    stop = threading.Event()

    def worker():
        while not stop.is_set():
            try:
                for ev in inputs.get_gamepad():
                    if stop.is_set():
                        return
                    code, val = ev.code, ev.state
                    if code in state["stick"]:
                        state["stick"][code] = val
                        state["events"] += 1
                    elif val == 1:
                        state["keys"].append(code)
                        state["events"] += 1
            except Exception:
                time.sleep(0.01)

    threading.Thread(target=worker, daemon=True).start()
    print(f"\n=== 读取 {args.seconds:.0f} 秒（请推摇杆 / 按 A、B）===\n")
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        time.sleep(0.4)
        st = state["stick"]
        dx = stick_to_delta(st["ABS_X"], scale=args.stick_scale)
        dy = -stick_to_delta(st["ABS_Y"], scale=args.stick_scale)
        dz = -stick_to_delta(st["ABS_RY"], scale=args.stick_scale)
        active = any(abs(v) > 1e-9 for v in (dx, dy, dz))
        print(
            f"  {'🔴 有效' if active else '  --   '} raw X={st['ABS_X']:+6d} Y={st['ABS_Y']:+6d} "
            f"RY={st['ABS_RY']:+6d} | dX={dx:+.4f} dY={dy:+.4f} dZ={dz:+.4f}"
        )
    stop.set()

    st = state["stick"]
    moved = any(abs(stick_to_delta(st[k], scale=args.stick_scale)) > 1e-9 for k in ("ABS_X", "ABS_Y", "ABS_RY"))
    print(f"\n=== 结果 ===")
    print(f"  收到事件数: {state['events']}")
    print(f"  按键: {state['keys'] if state['keys'] else '（无）'}")
    print(f"  摇杆最终值: X={st['ABS_X']} Y={st['ABS_Y']} RX={st['ABS_RX']} RY={st['ABS_RY']}")
    if moved:
        print("  ✅ 摇杆链路正常（real_collect 必须加 --controller gamepad）")
    elif state["events"] > 0:
        print("  ⚠️ 有事件但摇杆未过死区（死区 3000 ≈ 摇杆 9% 行程），推到底再试")
    else:
        print("  ❌ 完全没收到事件：检查手柄连接/权限/是否被其他程序独占")


if __name__ == "__main__":
    main()
