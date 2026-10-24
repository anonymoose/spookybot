"""Speak a line through the bot's own voice, speaker and jaw.

The running service holds the speaker, so stop it first (`make say` does the
stop / speak / restart for you from a dev machine):

    python -m spookybot.tools.say "Come closer. I don't bite. Much."
    echo "This is a test." | python -m spookybot.tools.say
"""
from __future__ import annotations

import logging
import sys
import time

from ..audio_io import Speaker
from ..config import CONFIG
from ..servos import ServoBus
from ..tts import Speech

DEFAULT_TEXT = "This is a test."


def main() -> int:
    logging.basicConfig(level=logging.INFO, datefmt="%H:%M:%S",
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    text = " ".join(sys.argv[1:]).strip()
    if not text and not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    text = text or DEFAULT_TEXT

    servos = ServoBus(CONFIG.servos)
    speaker = Speaker(CONFIG.audio, jaw_callback=servos.set_jaw)
    tts = Speech(CONFIG.tts, CONFIG.audio.playback_rate)

    t0 = time.monotonic()
    nbytes = 0
    speaker.start()
    try:
        for chunk in tts.stream(text):
            speaker.feed(chunk)
            nbytes += len(chunk)
    finally:
        speaker.done_feeding()
        finished = speaker.wait(timeout=30.0)
        speaker.close()
        servos.set_jaw(0.0)
        servos.close()

    seconds = nbytes / 2 / CONFIG.audio.playback_rate
    print(f"said {text!r}: {seconds:.1f}s of audio in {time.monotonic() - t0:.1f}s")
    if not nbytes:
        print("no audio came back from TTS", file=sys.stderr)
        return 1
    return 0 if finished else 1


if __name__ == "__main__":
    sys.exit(main())
