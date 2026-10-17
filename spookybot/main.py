"""Orchestrator: the state machine that ties camera, ears, brain and jaw together.

States
------
IDLE       nobody in frame; head at rest, occasional idle motion
GREET      someone just arrived; head speaks first (this is what makes people
           engage — a prop that waits to be spoken to mostly gets ignored)
LISTEN     mic open, waiting for an utterance
THINK      STT + first token from Claude
SPEAK      streaming TTS out the speaker, jaw driven by the audio buffer
           -> back to LISTEN for a follow-up, or IDLE on timeout

Without presence detection (no Hailo, or vision off) the head can't see anyone
arrive, so IDLE listens instead: a guest arrives by speaking, and GREET is
skipped. The first reply carries the costume still when the camera works.
"""
from __future__ import annotations

import argparse
import logging
import random
import signal
import sys
import threading
import time

from .audio_io import MicStream, Speaker
from .brain import Brain
from .config import CONFIG, Config
from .servos import ServoBus
from .stt import Transcriber
from .tts import Speech
from .vad import UtteranceDetector
from .vision import Vision

log = logging.getLogger("spookybot")

GREETINGS = [
    "Well. Someone's brave.",
    "Come closer. I don't bite. Much.",
    "Ah — a visitor. Say something.",
    "You there. Yes, you. Talk to me.",
    "I've been waiting all year for this.",
]

CONFUSED = [
    "Didn't catch that. Louder.",
    "Say again? It's loud in here, even for the dead.",
    "What was that?",
]


class SpookyBot:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.running = True
        self.state = "IDLE"
        self._last_greet = 0.0

        log.info("Initializing servos…")
        self.servos = ServoBus(cfg.servos)

        log.info("Initializing audio…")
        self.mic = MicStream(cfg.audio)
        self.speaker = Speaker(cfg.audio, jaw_callback=self.servos.set_jaw)
        self.detector = UtteranceDetector(cfg.vad, cfg.audio.sample_rate,
                                          cfg.audio.block_ms)

        log.info("Initializing vision…")
        self.vision = Vision(cfg.vision)

        log.info("Initializing brain…")
        self.brain = Brain(cfg.brain)
        self.stt = Transcriber(cfg.stt, cfg.audio.sample_rate)
        self.tts = Speech(cfg.tts, cfg.audio.playback_rate)

        self._idle_thread = threading.Thread(target=self._idle_motion, daemon=True)
        self._idle_thread.start()
        log.info("Ready.")

    # -- behaviors ----------------------------------------------------------

    def _idle_motion(self) -> None:
        """Small random head drift when idle. A perfectly still prop reads as
        broken; a slowly moving one reads as alive and pulls people over."""
        while self.running:
            time.sleep(random.uniform(3.0, 8.0))
            if self.state != "IDLE" or not self.cfg.servos.enabled:
                continue
            self.servos.set_normalized("pan", random.uniform(0.35, 0.65))
            self.servos.set_normalized("tilt", random.uniform(0.4, 0.6))

    def _track_guest(self) -> None:
        guest = self.vision.snapshot()
        if guest.present:
            self.servos.look_at(guest.center_x, guest.center_y)

    def say(self, sentences) -> bool:
        """Speak an iterable of sentences, streaming each through TTS."""
        self.state = "SPEAK"
        self.speaker.start()

        if self.cfg.audio.mute_mic_while_speaking:
            self.mic.gate(3600)  # gate generously; we lift it explicitly below

        spoke_anything = False
        try:
            for sentence in sentences:
                if not sentence.strip():
                    continue
                log.info("SAY: %s", sentence)
                for chunk in self.tts.stream(sentence):
                    self.speaker.feed(chunk)
                    spoke_anything = True
        finally:
            self.speaker.done_feeding()
            self.speaker.wait(timeout=30.0)
            self.speaker.close()
            self.servos.set_jaw(0.0)
            if self.cfg.audio.mute_mic_while_speaking:
                # Short tail so room reverb doesn't get transcribed as a guest.
                time.sleep(self.cfg.audio.mic_gate_tail_ms / 1000.0)
                self.mic.ungate()
                self.mic.flush()
        return spoke_anything

    def greet(self) -> None:
        self.state = "GREET"
        self._last_greet = time.monotonic()
        self.brain.reset()

        image = None
        note = None
        if self.cfg.vision.costume_vision:
            image = self.vision.capture_jpeg()
            if image:
                note = ("A guest just walked up. This is what your camera sees. "
                        "Greet them in one sentence and work in something "
                        "specific about their costume if you can identify it.")

        if image is not None:
            self.say(self.brain.respond("*a guest approaches*", image_jpeg=image,
                                        scene_note=note))
        else:
            self.say(iter([random.choice(GREETINGS)]))

    def converse_turn(self, audio=None, image=None, scene_note=None) -> bool:
        """One listen/think/speak cycle. Returns False when the guest is done.

        Pass `audio` to answer an utterance that was already captured, and
        `image`/`scene_note` to show the model the costume with this turn.
        """
        self.state = "LISTEN"
        self._track_guest()

        if audio is None:
            audio = self.detector.capture(
                self.mic,
                timeout_s=self.cfg.listen_timeout_s,
                on_speech_start=lambda: setattr(self, "state", "LISTEN"),
            )
            if audio is None:
                return False

        self.state = "THINK"
        try:
            text = self.stt.transcribe(audio)
        except Exception as exc:
            log.error("STT failed: %s", exc)
            self.say(iter([random.choice(CONFUSED)]))
            return True

        if not text or len(text) < 2:
            self.say(iter([random.choice(CONFUSED)]))
            return True

        log.info("HEARD: %s", text)
        self.say(self.brain.respond(text, image_jpeg=image, scene_note=scene_note))
        return True

    def start_by_voice(self) -> bool:
        """Voice-activated arrival, for when nothing can see guests walk up.

        Greeting on a timer instead would talk to an empty room all night, so
        wait for someone to speak and answer them. Returns True if a guest
        spoke and a conversation started.
        """
        self.state = "IDLE"
        audio = self.detector.capture(
            self.mic,
            timeout_s=self.cfg.listen_timeout_s,
            on_speech_start=lambda: setattr(self, "state", "LISTEN"),
        )
        if audio is None:
            return False

        self.brain.reset()
        image = note = None
        if self.cfg.vision.costume_vision:
            image = self.vision.capture_jpeg()
            if image:
                note = ("A guest just walked up and spoke to you. This is what "
                        "your camera sees. Answer them, and work in something "
                        "specific about their costume if you can identify it.")
        self.converse_turn(audio=audio, image=image, scene_note=note)
        return True

    # -- main loop ----------------------------------------------------------

    def run(self) -> None:
        detecting = self.vision.detecting
        if not detecting:
            log.info("No presence detection — voice-activated mode: guests "
                     "start a conversation by speaking")
        while self.running:
            try:
                if detecting:
                    if not self.vision.snapshot().present:
                        self.state = "IDLE"
                        time.sleep(0.2)
                        continue

                    cooled = (time.monotonic() - self._last_greet
                              > self.cfg.greet_cooldown_s)
                    if cooled:
                        # Speaking first is what gets people to engage; a prop
                        # that waits to be addressed mostly gets looked at and
                        # ignored.
                        self.greet()
                elif not self.start_by_voice():
                    continue

                while self.running and self.converse_turn():
                    if detecting and not self.vision.snapshot().present:
                        log.info("Guest left mid-conversation")
                        break

                self.state = "IDLE"
                self.servos.rest()
                self.brain.reset()
            except KeyboardInterrupt:
                break
            except Exception as exc:
                log.exception("loop error: %s", exc)
                time.sleep(1.0)

    def shutdown(self) -> None:
        # SIGTERM and the `finally` block both land here; closing a stream
        # twice raises and would mask the real exit path.
        if getattr(self, "_shut_down", False):
            return
        self._shut_down = True

        log.info("Shutting down…")
        self.running = False
        for step in (self.speaker.stop, self.mic.close, self.vision.close,
                     self.servos.close):
            try:
                step()
            except Exception as exc:
                log.debug("shutdown step %s failed: %s", step.__name__, exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Halloween talking head")
    parser.add_argument("--no-vision", action="store_true")
    parser.add_argument("--no-servos", action="store_true")
    parser.add_argument("--say", help="Speak one line and exit (smoke test)")
    parser.add_argument("--log-level", default=CONFIG.log_level)
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = CONFIG
    if args.no_vision:
        cfg.vision.enabled = False
    if args.no_servos:
        cfg.servos.enabled = False

    bot = SpookyBot(cfg)
    signal.signal(signal.SIGTERM, lambda *_: bot.shutdown())

    try:
        if args.say:
            bot.say(iter([args.say]))
        else:
            bot.run()
    except KeyboardInterrupt:
        pass
    finally:
        bot.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
