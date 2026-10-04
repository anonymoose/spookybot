"""Measure the room's noise floor and suggest an RMS threshold.

Run this TWICE, and it is the single highest-value tuning step in the project:

  1. In the empty room  -> tells you the floor
  2. During the party, music on, people talking, standing where guests will
     stand -> tells you the real floor

Use the party number. A threshold tuned in a silent garage will make the head
respond to the whole room on Halloween night.

    python -m spookybot.tools.calibrate_noise --seconds 15
"""
from __future__ import annotations

import argparse
import time

from ..config import CONFIG
from ..audio_io import MicStream
from ..vad import measure_noise_floor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--speak-test", action="store_true",
                        help="Also measure YOU talking at normal distance")
    args = parser.parse_args()

    mic = MicStream(CONFIG.audio)
    try:
        print(f"Measuring ambient noise for {args.seconds:.0f}s — "
              "stay quiet, let the room be the room…")
        time.sleep(0.5)
        mic.flush()
        ambient = measure_noise_floor(mic, args.seconds)

        print("\n-- ambient --")
        for key in ("median", "p90", "p99", "max"):
            print(f"  {key:>8}: {ambient[key]:9.1f}")

        floor = ambient["suggested_floor"]

        if args.speak_test:
            input("\nNow stand where guests will stand and press Enter, "
                  "then talk normally to the prop…")
            mic.flush()
            speech = measure_noise_floor(mic, 8.0)
            print("\n-- speech --")
            for key in ("median", "p90", "p99", "max"):
                print(f"  {key:>8}: {speech[key]:9.1f}")

            # Put the threshold between the two distributions, biased toward
            # the ambient side so quiet guests still get through.
            gap = speech["median"] - ambient["p90"]
            if gap <= 0:
                print("\n!! Your voice is not louder than the room at that "
                      "distance. Move the mic closer to where guests stand, "
                      "or turn the music down. No threshold will fix this.")
            else:
                floor = ambient["p90"] + gap * 0.35
                print(f"\n  separation: {gap:.1f} — good" if gap > 300
                      else f"\n  separation: {gap:.1f} — tight, expect misses")

        print(f"\nSet this:\n\n  export SB_RMS_FLOOR={floor:.0f}\n")
        return 0
    finally:
        mic.close()


if __name__ == "__main__":
    raise SystemExit(main())
