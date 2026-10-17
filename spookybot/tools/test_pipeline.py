"""Exercise each subsystem one at a time, so a failure tells you WHICH part
is broken instead of just "it doesn't work".

    python -m spookybot.tools.test_pipeline --all
    python -m spookybot.tools.test_pipeline --speaker --jaw
"""
from __future__ import annotations

import argparse
import logging
import time

from ..config import CONFIG


def test_speaker() -> bool:
    from ..audio_io import Speaker
    from ..tts import Speech
    print("\n[speaker] Speaking a test line…")
    try:
        speaker = Speaker(CONFIG.audio)
        tts = Speech(CONFIG.tts, CONFIG.audio.playback_rate)
        speaker.start()
        got = 0
        for chunk in tts.stream("Testing, one two three. If you can hear this "
                                "clearly from across the room, your speaker is "
                                "loud enough."):
            speaker.feed(chunk)
            got += len(chunk)
        speaker.done_feeding()
        speaker.wait(timeout=30)
        speaker.close()
        print(f"[speaker] OK — {got/1024:.0f} KB of audio played")
        return got > 0
    except Exception as exc:
        print(f"[speaker] FAILED: {exc}")
        return False


def test_mic() -> bool:
    from ..audio_io import MicStream
    from ..vad import UtteranceDetector
    from ..stt import Transcriber
    print("\n[mic] Say something in the next 10 seconds…")
    mic = None
    try:
        mic = MicStream(CONFIG.audio)
        detector = UtteranceDetector(CONFIG.vad, CONFIG.audio.sample_rate,
                                     CONFIG.audio.block_ms)
        audio = detector.capture(mic, timeout_s=10.0)
        if audio is None:
            print("[mic] FAILED: no utterance detected. Either the mic isn't "
                  "capturing, or SB_RMS_FLOOR is too high. Run calibrate_noise.")
            return False
        print(f"[mic] Captured {len(audio)/CONFIG.audio.sample_rate:.1f}s")
        text = Transcriber(CONFIG.stt, CONFIG.audio.sample_rate).transcribe(audio)
        print(f"[mic] Transcribed: {text!r}")
        return bool(text)
    except Exception as exc:
        print(f"[mic] FAILED: {exc}")
        return False
    finally:
        if mic:
            mic.close()


def test_brain() -> bool:
    from ..brain import Brain
    print("\n[brain] Asking Claude for a line…")
    try:
        brain = Brain(CONFIG.brain)
        t0 = time.perf_counter()
        first = None
        out = []
        for sentence in brain.respond("Hello there, what are you?"):
            if first is None:
                first = time.perf_counter() - t0
            out.append(sentence)
        print(f"[brain] First sentence in {first:.2f}s: {out[0] if out else '—'}")
        print(f"[brain] Full: {' '.join(out)}")
        return bool(out)
    except Exception as exc:
        print(f"[brain] FAILED: {exc}")
        return False


def test_vision() -> bool:
    from ..vision import Vision
    print("\n[vision] Watching for a person for 10 seconds — stand in frame…")
    vision = None
    try:
        vision = Vision(CONFIG.vision)
        if vision._picam is None:
            print("[vision] FAILED: camera not initialized")
            return False
        if not vision.detecting:
            print("[vision] Camera-only mode (no Hailo): costume stills work, "
                  "person detection is off. Check `lspci | grep -i hailo`.")
            jpeg = vision.capture_jpeg()
            print(f"[vision] Snapshot: {len(jpeg)/1024:.0f} KB"
                  if jpeg else "[vision] Snapshot failed")
            return bool(jpeg)
        seen = False
        for _ in range(100):
            guest = vision.snapshot()
            if guest.present:
                print(f"[vision] Person detected — count={guest.count} "
                      f"center=({guest.center_x:.2f}, {guest.center_y:.2f}) "
                      f"height={guest.box_height:.2f}")
                seen = True
                break
            time.sleep(0.1)
        if not seen:
            print("[vision] No person detected. Check framing, or lower "
                  "SB_MIN_BOX_HEIGHT / SB_DETECT_THRESHOLD.")
            return False
        jpeg = vision.capture_jpeg()
        print(f"[vision] Snapshot: {len(jpeg)/1024:.0f} KB"
              if jpeg else "[vision] Snapshot failed")
        return True
    except Exception as exc:
        print(f"[vision] FAILED: {exc}")
        return False
    finally:
        if vision:
            vision.close()


def test_jaw() -> bool:
    """Play a sweep and watch the jaw. This is the one you tune by eye."""
    from ..audio_io import Speaker
    from ..servos import ServoBus
    from ..tts import Speech
    print("\n[jaw] Speaking while driving the jaw — watch the mouth…")
    servos = None
    try:
        servos = ServoBus(CONFIG.servos)
        speaker = Speaker(CONFIG.audio, jaw_callback=servos.set_jaw)
        tts = Speech(CONFIG.tts, CONFIG.audio.playback_rate)
        speaker.start()
        for chunk in tts.stream(
            "Watch my mouth now. Does it open on the loud parts and close "
            "between words? If it barely moves, raise the jaw range. If it "
            "hangs open, lower it."
        ):
            speaker.feed(chunk)
        speaker.done_feeding()
        speaker.wait(timeout=30)
        speaker.close()
        servos.set_jaw(0.0)
        time.sleep(0.5)
        print("[jaw] Done. Tune ServoConfig.jaw min/max and SB_JAW_ALPHA.")
        return True
    except Exception as exc:
        print(f"[jaw] FAILED: {exc}")
        return False
    finally:
        if servos:
            servos.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("all", "speaker", "mic", "brain", "vision", "jaw"):
        parser.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(levelname)-7s %(name)s: %(message)s")

    tests = [("speaker", test_speaker), ("mic", test_mic), ("brain", test_brain),
             ("vision", test_vision), ("jaw", test_jaw)]
    selected = [(n, f) for n, f in tests if args.all or getattr(args, n)]
    if not selected:
        parser.print_help()
        return 1

    results = {name: fn() for name, fn in selected}
    print("\n" + "=" * 40)
    for name, ok in results.items():
        print(f"  {name:>8}: {'PASS' if ok else 'FAIL'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
