"""Interactively find each servo's safe travel limits.

Do this BEFORE the servo is bolted to anything. Drive a jaw servo past the
mechanical stop of a skull and you will strip the gears, burn the servo, or
crack the jaw — usually within a couple of seconds, because a stalled servo
keeps pushing at full torque.

    python -m spookybot.tools.servo_jog --servo jaw

Keys: j / k  = -1 / +1 degree     h / l = -5 / +5 degrees
      s      = record this angle  q     = quit
"""
from __future__ import annotations

import argparse
import sys
import termios
import tty

from ..config import CONFIG
from ..servos import ServoBus, _HAVE_HW


def getch() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--servo", choices=["jaw", "pan", "tilt"], default="jaw")
    parser.add_argument("--start", type=float, default=90.0)
    args = parser.parse_args()

    if not _HAVE_HW:
        print("No PCA9685 libraries found — this tool needs real hardware.")
        return 1

    # Full range while jogging, so the limits themselves don't clamp you.
    cfg = CONFIG.servos
    limits = {"jaw": cfg.jaw, "pan": cfg.pan, "tilt": cfg.tilt}[args.servo]
    limits.min_deg, limits.max_deg = 0.0, 180.0
    cfg.track_alpha = cfg.jaw_alpha = 0.35

    bus = ServoBus(cfg)
    angle = args.start
    recorded: list[float] = []

    print(f"Jogging {args.servo}. j/k = ±1°, h/l = ±5°, s = record, q = quit")
    try:
        while True:
            bus.set_normalized(args.servo, angle / 180.0)
            print(f"\r  {args.servo}: {angle:6.1f}°   recorded={recorded}   ",
                  end="", flush=True)
            key = getch().lower()
            if key == "q":
                break
            elif key == "j":
                angle -= 1
            elif key == "k":
                angle += 1
            elif key == "h":
                angle -= 5
            elif key == "l":
                angle += 5
            elif key == "s":
                recorded.append(round(angle, 1))
            angle = max(0.0, min(180.0, angle))
    finally:
        bus.close()

    print("\n")
    if len(recorded) >= 2:
        lo, hi = min(recorded), max(recorded)
        if args.servo == "jaw":
            print("Put this in .env (closed a degree or two short of where the "
                  "jaw physically shuts, or it will grind):\n")
            print(f"    SB_JAW_CLOSED_DEG={lo:g}\n    SB_JAW_OPEN_DEG={hi:g}\n")
            print("If open and closed come out backwards, swap the two numbers.")
            return 0
        print(f"Put this in config.py under ServoConfig.{args.servo}:\n")
        rest = lo if args.servo == "jaw" else (lo + hi) / 2
        print(f"    {args.servo}: ServoLimits = field(default_factory=lambda: "
              f"ServoLimits(\n        channel={limits.channel}, min_deg={lo}, "
              f"max_deg={hi}, rest_deg={rest}))\n")
        print("For the jaw, min_deg should be mouth CLOSED and max_deg mouth "
              "OPEN.\nIf it comes out backwards, set invert=True instead of "
              "swapping the numbers.")
    else:
        print("Record at least two angles (press 's') to get limits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
