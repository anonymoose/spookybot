"""List every audio device the Pi can see, so you can set SB_INPUT_DEVICE
and SB_OUTPUT_DEVICE to a substring that matches.

    python -m spookybot.tools.list_devices
"""
from __future__ import annotations

import sounddevice as sd


def main() -> int:
    print(f"{'idx':>4}  {'in':>3} {'out':>3}  {'rate':>7}  name")
    print("-" * 72)
    for idx, dev in enumerate(sd.query_devices()):
        print(f"{idx:>4}  {dev['max_input_channels']:>3} "
              f"{dev['max_output_channels']:>3}  "
              f"{int(dev['default_samplerate']):>7}  {dev['name']}")

    print()
    try:
        din, dout = sd.default.device
        print(f"System default input : {din}")
        print(f"System default output: {dout}")
    except Exception:
        pass
    print("\nSet SB_INPUT_DEVICE / SB_OUTPUT_DEVICE to any unique substring "
          "of the name you want.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
