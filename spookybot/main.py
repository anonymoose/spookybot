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
import itertools
import logging
import random
import re
import signal
import sys
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from . import net
from .audio_io import MicStream, Speaker
from .brain import Brain, split_sentences
from .config import CONFIG, Config
from .preload import Preloader
from .servos import ServoBus
from .stt import EarlyTranscript, Transcriber
from .tts import Speech
from .vad import UtteranceDetector
from .vision import Vision
from .wake import strip_wake_phrase

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


# Played the instant a guest finishes a question Claude has to answer, while
# it writes the reply: what a person says while they think. Voiced at startup.
FILLERS = ("Hmm.", "Hmm, let me see.", "Well, well.", "Ooh.", "Let me think.",
           "Oh, now.")

# A guest taking their leave ends the conversation with a goodbye, the way
# "quiet bones" does, instead of waiting out the listen timeout.
GOODBYE_RE = re.compile(r"\b(?:good ?bye|bye|good ?night|see (?:you|ya)|gotta go|"
                        r"got to go|have to go|nice (?:talking|chatting|meeting))\b",
                        re.IGNORECASE)
GOODBYE_NOTE = ("The guest is saying goodbye and leaving. Say goodbye in ONE "
                "short sentence, in character.")

# Said to the room after SB_LONELY_AFTER_S with no conversation, written fresh
# each time (one topic picked at random) rather than prepared.
LONELY_TOPICS = ("how lonely you are up on this wall",
                 "how nobody is talking to you",
                 "how you wish somebody would bring you a drink")
LONELY_USER = "*nobody has talked to you for half an hour*"
LONELY_NOTE = ("Nobody has talked to you for a long while. Say ONE short sentence, "
               "under 15 words, out loud to whoever might be nearby, about {topic}. "
               "Dry and funny, in character. It is spoken aloud, so no stage "
               "directions, and leave your backstory out of it.")

# Any request that mentions a joke gets a prepared one ("joke about cats" too).
JOKE_RE = re.compile(r"\bjokes?\b", re.IGNORECASE)

INTRO_USER = "*a guest says hello to you*"
# Kept short on purpose: guests who said hello want to talk, not listen. At
# ~35 words the intros ran ~12 s; ~12 words is ~5 s.
INTRO_NOTE = ("A guest just said your name to get your attention. Introduce "
              "yourself in TWO short sentences, about 12 words in total and "
              "never more than 15 (roughly 4 seconds spoken): you are Bones, "
              "bolted to this wall since 1874, and you offer to judge their "
              "costume. Example: \"I'm Bones, bolted to this wall since 1874. "
              "Shall I judge your costume?\" Word it differently every time.")
JOKE_USER = "Tell me a joke."
JOKE_NOTE = ("A guest asked for a joke. Tell ONE short, clean, PG-13 joke in "
             "your voice: exactly two sentences, a setup and a punchline, under "
             "30 words in total. It is spoken aloud, so no stage directions. "
             "Make it a different joke every time.")
# Without a nudge every restart opens with the same skeleton-at-a-party joke.
JOKE_THEMES = ("skeletons and bones", "pumpkins", "ghosts", "vampires", "witches",
               "mummies", "zombies", "candy and trick-or-treating", "costumes",
               "black cats", "spiders", "haunted houses", "Florida in October",
               "cemeteries", "werewolves", "being a head on a wall")


# Questions with a fixed answer in persona.md's "Party Questions", prepared ahead
# like the jokes. Ken and Amy also need a where-type word, so a guest who says
# "I'm Ken" isn't told he's off getting a hotdog.
BATHROOM_RE = re.compile(r"\b(?:bathrooms?|restrooms?|toilets?|loo|washroom|pee)\b",
                         re.IGNORECASE)
WHERE_RE = re.compile(r"\b(?:where|find|seen|looking|hiding|locate)\b", re.IGNORECASE)
NAME_RES = {"ken": re.compile(r"\bken\b", re.IGNORECASE),
            "amy": re.compile(r"\bamy\b", re.IGNORECASE)}
DIRECTIONS = {"bathroom": "Where is the bathroom?",
              "ken": "Where is Ken?",
              "amy": "Where is Amy?"}
DIRECTIONS_NOTE = ("A guest asked this. Answer with the answer in your Party "
                   "Questions notes, keeping its facts exactly, in your own "
                   "voice: ONE or TWO short sentences, under 30 words in total. "
                   "Add no facts of your own (not even whether they are at the "
                   "party). "
                   "It is spoken aloud, so no stage directions. Word it "
                   "differently every time.")


def directions_asked(text: str) -> list:
    """The prepared-answer kinds a guest's words ask for, e.g. ["ken", "amy"]."""
    kinds = ["bathroom"] if BATHROOM_RE.search(text) else []
    if WHERE_RE.search(text):
        kinds += [kind for kind, name in NAME_RES.items() if name.search(text)]
    return kinds


def directions_request(kind: str):
    def request(avoid: list) -> tuple:
        note = DIRECTIONS_NOTE
        if avoid:
            note += " Do not reuse these earlier answers: " + " | ".join(avoid) + "."
        return DIRECTIONS[kind], note
    return request


def intro_request(avoid: list) -> tuple:
    note = INTRO_NOTE
    if avoid:
        note += " Do not reuse these earlier intros: " + " | ".join(avoid) + "."
    return INTRO_USER, note


def joke_request(avoid: list) -> tuple:
    note = f"{JOKE_NOTE} Theme this time: {random.choice(JOKE_THEMES)}."
    if avoid:
        note += " Do not reuse these jokes: " + " | ".join(avoid) + "."
    return JOKE_USER, note


def eastern_now() -> datetime:
    return datetime.now(ZoneInfo("America/New_York"))


def within_hours(hours: str, now: datetime) -> bool:
    """Whether `now` falls in "START-END" (24-hour clock, END exclusive). A
    window that wraps midnight ("18-1") works; a bad setting allows any time."""
    try:
        start, end = (int(h) % 24 for h in hours.split("-"))
    except ValueError:
        return True
    if start <= end:
        return start <= now.hour < end
    return now.hour >= start or now.hour < end


def spoken_time(now: datetime | None = None) -> str:
    """Eastern wall-clock time as a line for the speech synthesizer."""
    now = now or eastern_now()
    return f"{now.hour % 12 or 12}:{now.minute:02d} {'AM' if now.hour < 12 else 'PM'}"


class SpookyBot:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.running = True
        self._state = "IDLE"
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
        self.vision.status_fn = lambda: (f"state {self.state}, "
                                         f"eyes {self.servos.eye_mode}")

        log.info("Initializing brain…")
        self.brain = Brain(cfg.brain)
        # While Claude searches the web the voice goes quiet: flicker the eyes
        # (it is waiting on Anthropic), and go steady again once it talks.
        self._voice_started = False
        self.brain.on_wait = self._on_brain_wait
        self.stt = Transcriber(cfg.stt, cfg.audio.sample_rate)
        self.tts = Speech(cfg.tts, cfg.audio.playback_rate)
        self.preloader = None
        targets = {"intro": cfg.preload_intros, "joke": cfg.preload_jokes}
        prompts = {"intro": intro_request, "joke": joke_request}
        for kind in DIRECTIONS:
            targets[kind] = cfg.preload_directions
            prompts[kind] = directions_request(kind)
        if any(n > 0 for n in targets.values()):
            self.preloader = Preloader(
                self.brain, self.tts, targets, prompts,
                idle=lambda: self.state in ("IDLE", "LISTEN"))
            # Prepared answers carry the old persona's facts (where Ken is, the
            # bathroom...), so a new persona.md replaces them all.
            self.brain.on_persona_change = self.preloader.clear

        self._last_filler = None
        # When the current conversation started, and when the last one did (or
        # startup): the empty-frame check and the lonely timer count from these.
        self._session_start = self._last_session = time.monotonic()

        self._idle_thread = threading.Thread(target=self._idle_motion, daemon=True)
        self._idle_thread.start()
        if cfg.keep_warm_s > 0:
            threading.Thread(target=self._keep_warm, daemon=True,
                             name="keep-warm").start()
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

    def _keep_warm(self) -> None:
        """Keep the connections to the speech server and Anthropic open (see
        net.py): a reused connection answers in ~0.3 s, a fresh one in ~6 s."""
        last_claude = 0.0
        while self.running:
            time.sleep(net.SPEECH_PING_S)
            for name, ping in (("speech-to-text", self.stt.ping),
                               ("text-to-speech", self.tts.ping)):
                try:
                    ping()
                except Exception as exc:
                    log.debug("keep-warm %s ping failed: %s", name, exc)
            if time.monotonic() - last_claude >= self.cfg.keep_warm_s:
                last_claude = time.monotonic()
                try:
                    self.brain.ping()
                except Exception as exc:
                    log.debug("keep-warm Anthropic ping failed: %s", exc)

    def _early_transcript(self) -> EarlyTranscript:
        vad, audio = self.cfg.vad, self.cfg.audio
        block = audio.sample_rate * audio.block_ms // 1000
        return EarlyTranscript(self.stt, (vad.end_frames - vad.pause_frames) * block)

    def prepare_fillers(self) -> None:
        ready = sum(self.tts.keep(line) for line in FILLERS)
        log.info("VOICE %d of %d fillers ready", ready, len(FILLERS))

    def _filler(self) -> str | None:
        """A filler to play while Claude answers, or None (by chance, or none
        ready). Never the same one twice in a row."""
        if random.random() >= self.cfg.filler_chance:
            return None
        ready = [f for f in FILLERS if f != self._last_filler and self.tts.is_kept(f)]
        if not ready:
            return None
        self._last_filler = random.choice(ready)
        return self._last_filler

    def _start_session(self) -> None:
        self._session_start = self._last_session = time.monotonic()

    def _nobody_here(self) -> bool:
        """True once nobody at all has been in frame for SB_EMPTY_FRAME_END_S,
        counted from the start of this conversation. Never true without a
        working person detector: then nobody can be seen leaving."""
        limit = self.cfg.empty_frame_end_s
        if limit <= 0 or not self.vision.detecting:
            return False
        seen = max(self.vision.snapshot().anyone_seen, self._session_start)
        return time.monotonic() - seen >= limit

    def maybe_lonely(self) -> bool:
        """Say one fresh line to the room if nobody has started a conversation
        for SB_LONELY_AFTER_S, during SB_LONELY_HOURS (Eastern). Returns True
        if it spoke."""
        if (self.cfg.lonely_after_s <= 0
                or time.monotonic() - self._last_session < self.cfg.lonely_after_s
                or not within_hours(self.cfg.lonely_hours, eastern_now())):
            return False
        self._last_session = time.monotonic()      # at most once per period
        topic = random.choice(LONELY_TOPICS)
        text = self.brain.generate(LONELY_USER, LONELY_NOTE.format(topic=topic),
                                   max_tokens=80)
        if not text:
            return False
        log.info("VOICE lonely (%s): %s", topic, text)
        self.say(iter(split_sentences(text)))
        return True

    def _track_guest(self) -> None:
        guest = self.vision.snapshot()
        if guest.present:
            self.servos.look_at(guest.center_x, guest.center_y)

    def say(self, sentences) -> bool:
        """Speak an iterable of sentences, streaming each through TTS."""
        self.state = "SPEAK"
        self._voice_started = False
        self.speaker.start()

        if self.cfg.audio.mute_mic_while_speaking:
            self.mic.gate(3600)  # gate generously; we lift it explicitly below

        spoke_anything = False
        try:
            for sentence in sentences:
                if not sentence.strip():
                    continue
                log.info("VOICE SAY: %s", sentence)
                for chunk in self.tts.stream(sentence):
                    self.speaker.feed(chunk)
                    if not spoke_anything:
                        self._voice_started = True
                        self.servos.set_eye_mode("steady")
                    spoke_anything = True
        finally:
            self.speaker.done_feeding()
            self.speaker.wait(timeout=30.0)
            self.speaker.close()
            self.servos.set_jaw(0.0)
            self.servos.set_eye_mode("pulse")
            if self.cfg.audio.mute_mic_while_speaking:
                # Short tail so room reverb doesn't get transcribed as a guest.
                time.sleep(self.cfg.audio.mic_gate_tail_ms / 1000.0)
                self.mic.ungate()
                self.mic.flush()
        return spoke_anything

    def greet(self) -> None:
        self.state = "GREET"
        self._last_greet = time.monotonic()
        self._start_session()
        self.brain.reset()

        image = None
        note = None
        if self.cfg.vision.costume_vision:
            image = self.vision.capture_jpeg()
            if image:
                note = ("A guest just walked up. This is what your camera sees. "
                        "Greet them in one sentence and work in something "
                        "specific about their costume if you can identify it.")
                seen = self.brain.recent_greetings
                if seen:
                    note += (" Lines you used on earlier arrivals tonight: "
                             + " | ".join(seen) + ". If this costume matches "
                             "one of those, say you've already seen it tonight "
                             "instead of greeting it as new.")

        if image is not None:
            self.say(self._remembered(self.brain.respond(
                "*a guest approaches*", image_jpeg=image, scene_note=note)))
        else:
            self.say(iter([random.choice(GREETINGS)]))

    def _remembered(self, sentences):
        """Pass sentences through, then file what was said under recent greetings."""
        lines: list[str] = []
        for sentence in sentences:
            lines.append(sentence)
            yield sentence
        if lines and self.brain.recent_greetings.maxlen:
            self.brain.recent_greetings.append(" ".join(lines))

    def converse_turn(self, audio=None, image=None, scene_note=None,
                      text=None) -> bool:
        """One listen/think/speak cycle. Returns False when the guest is done.

        Pass `audio` to answer an utterance that was already captured, and
        `image`/`scene_note` to show the model the costume with this turn. Pass
        `text` as well to skip transcribing that audio.
        """
        self.state = "LISTEN"
        self._track_guest()

        early = None
        if audio is None:
            if self._nobody_here():
                log.info("VISION nobody in frame for %.0f s: ending the conversation",
                         self.cfg.empty_frame_end_s)
                return False
            early = self._early_transcript()
            audio = self.detector.capture(
                self.mic,
                timeout_s=self.cfg.listen_timeout_s,
                on_speech_start=lambda: setattr(self, "state", "LISTEN"),
                on_pause=early.on_pause,
                should_stop=self._nobody_here,
            )
            if audio is None:
                if self._nobody_here():
                    log.info("VISION nobody in frame for %.0f s: ending the "
                             "conversation", self.cfg.empty_frame_end_s)
                return False

        self.state = "THINK"
        if text is None:
            try:
                text = early.result(audio) if early else self.stt.transcribe(audio)
            except Exception as exc:
                log.error("VOICE STT failed: %s", exc)
                self.say(iter([random.choice(CONFUSED)]))
                return True

        if not text or len(text) < 2:
            self.say(iter([random.choice(CONFUSED)]))
            return True

        log.info("VOICE HEARD: %s", text)
        if (self.cfg.quiet_phrase
                and strip_wake_phrase(text, self.cfg.quiet_phrase) is not None):
            # The off-switch for a chat that has gone on too long. Returning
            # False sends run() back to IDLE, which waits for the wake phrase.
            log.info("VOICE QUIET %r: ending the conversation", self.cfg.quiet_phrase)
            self.say(iter(["As you wish."]))
            return False
        if GOODBYE_RE.search(text):
            log.info("VOICE goodbye: ending the conversation")
            self.say(self.brain.respond(text, scene_note=GOODBYE_NOTE))
            return False
        if self.preloader is not None and JOKE_RE.search(text):
            prepared = self.preloader.take("joke")
            if prepared:
                log.info("VOICE joke from the preloaded queue")
                self.say(iter(prepared))
                self.brain.inject_turn(text, " ".join(prepared))
                return True
        if self._answer_directions(text):
            return True
        # A fresh still with every turn, so "what do you think of my costume?"
        # works at any point in the chat, not just the first reply. The brain
        # keeps only the newest image in context, so this costs one image per
        # call and no more.
        if image is None and self.cfg.vision.costume_vision:
            image = self.vision.capture_jpeg()
        # The reply is a generator: nothing is sent to Claude until say() asks
        # for its first sentence, which is after the filler has started playing.
        sentences = self.brain.respond(text, image_jpeg=image, scene_note=scene_note)
        filler = self._filler()
        if filler:
            log.info("VOICE filler %r while Claude answers", filler)
            sentences = itertools.chain([filler], sentences)
        self.say(sentences)
        return True

    def _answer_directions(self, text: str) -> bool:
        """Play prepared answers for the bathroom / Ken / Amy, if every one the
        guest asked for is ready ("Where are Ken and Amy?" plays both). If any
        is missing, take none, so Claude answers the whole question live."""
        if self.preloader is None:
            return False
        kinds = directions_asked(text)
        counts = self.preloader.counts()
        if not kinds or not all(counts.get(kind, 0) > 0 for kind in kinds):
            return False
        sentences = [s for kind in kinds for s in self.preloader.take(kind)]
        log.info("VOICE %s from the preloaded queue", " + ".join(kinds))
        self.say(iter(sentences))
        self.brain.inject_turn(text, " ".join(sentences))
        return True

    def start_by_voice(self) -> bool:
        """Voice-activated arrival, for when nothing can see guests walk up.

        Greeting on a timer instead would talk to an empty room all night, so
        wait for someone to speak and answer them. Returns True if a guest
        spoke and a conversation started.
        """
        self.state = "IDLE"
        early = self._early_transcript()
        audio = self.detector.capture(
            self.mic,
            timeout_s=self.cfg.listen_timeout_s,
            on_speech_start=lambda: setattr(self, "state", "LISTEN"),
            on_pause=early.on_pause if self.cfg.wake_phrase else None,
        )
        if audio is None:
            return False

        if self.cfg.wake_phrase:
            # Everything said in the room gets transcribed here; only the wake
            # phrase starts a conversation.
            try:
                heard = early.result(audio)
            except Exception as exc:
                log.error("VOICE STT failed: %s", exc)
                return False
            rest = strip_wake_phrase(heard or "", self.cfg.wake_phrase)
            if rest is None:
                log.info("VOICE ignored (no %r): %s", self.cfg.wake_phrase, heard)
                return False
            log.info("VOICE WAKE %r (then: %r)", self.cfg.wake_phrase, rest)
            self.introduce()
            return True

        self.brain.reset()
        self._start_session()
        image = note = None
        if self.cfg.vision.costume_vision:
            image = self.vision.capture_jpeg()
            if image:
                note = ("A guest just walked up and spoke to you. This is what "
                        "your camera sees. Answer them, and work in something "
                        "specific about their costume if you can identify it.")
        self.converse_turn(audio=audio, image=image, scene_note=note)
        return True

    def introduce(self) -> None:
        """Answer the wake phrase with a fresh, short intro, then listen.

        Whatever followed the wake phrase is dropped: the intro asks a question,
        so the guest's next utterance is the real one.
        """
        self.state = "GREET"
        self.brain.reset()
        self._start_session()
        prepared = self.preloader.take("intro") if self.preloader else None
        if prepared:
            log.info("VOICE intro from the preloaded queue")
            self.say(iter(prepared))
            said = " ".join(prepared)
            self.brain.inject_turn(INTRO_USER, said)
            if self.brain.recent_greetings.maxlen:
                self.brain.recent_greetings.append(said)
        else:
            user_text, note = intro_request(list(self.brain.recent_greetings))
            self.say(self._remembered(
                self.brain.respond(user_text, scene_note=note)))

        self.converse_turn(scene_note=(
            "The guest is answering your intro; if they want their costume "
            "judged, be specific about what you see."))

    # -- main loop ----------------------------------------------------------

    # What the eyes do in each state: slow pulse while waiting for a guest, fast
    # flicker while waiting on Claude (the intro, or STT then Claude), steady
    # while the voice is playing. say() switches to steady on the first audio.
    _EYES = {"IDLE": "pulse", "LISTEN": "pulse", "THINK": "flicker",
             "GREET": "flicker", "SPEAK": "flicker"}

    @property
    def state(self) -> str:
        return self._state

    @state.setter
    def state(self, value: str) -> None:
        self._state = value
        servos = getattr(self, "servos", None)
        if servos is not None:
            servos.set_eye_mode(self._EYES.get(value, "pulse"))

    def _on_brain_wait(self, waiting: bool) -> None:
        if waiting:
            self.servos.set_eye_mode("flicker")
        elif self._voice_started:
            self.servos.set_eye_mode("steady")

    def _speak_line(self, text: str) -> None:
        try:
            self.say(iter([text]))
        except Exception as exc:
            log.warning("VOICE startup line failed (%r): %s", text, exc)

    def announce_ready(self) -> None:
        """Say hello and the time once everything is up, so whoever restarted the
        bot knows it is alive and the speaker and clock are right."""
        self._speak_line(f"Hello, my name is Bones. It is {spoken_time()} "
                         "Eastern time.")

    def startup_sequence(self) -> None:
        """Announce, stock the preload queues, announce again when it is ready.

        With announcements on, the stocking runs here, on the main thread, so
        nothing can talk over the announcements and the head is not listening
        until it says it is ready. With them off (a quick dev restart) the
        queues fill in the background and it listens straight away.
        """
        if self.cfg.announce_startup:
            self.announce_ready()
            if self.cfg.filler_chance > 0:
                self.prepare_fillers()
            if self.preloader is not None:
                self._speak_line("Preloading interactions.")
                self.servos.set_eye_mode("flicker")   # waiting on Claude
                t0 = time.monotonic()
                full = self.preloader.fill_initial()
                log.info("PRELOAD done in %.0f s (%s)", time.monotonic() - t0,
                         "queues full" if full else "incomplete, will keep "
                         "trying; replies not ready are written live")
                self.servos.set_eye_mode("pulse")
            self._speak_line("Bones is ready.")
        elif self.cfg.filler_chance > 0:
            threading.Thread(target=self.prepare_fillers, daemon=True).start()
        if self.preloader is not None:
            self.preloader.start()

    def run(self) -> None:
        self.startup_sequence()
        # With a wake phrase the camera still tracks guests, but only speech
        # starts a conversation.
        detecting = self.vision.detecting and not self.cfg.wake_phrase
        if self.cfg.wake_phrase:
            log.info("VOICE wake-phrase mode: say %r to start a conversation",
                     self.cfg.wake_phrase)
        elif not detecting:
            log.info("VISION no presence detection — voice-activated mode: guests "
                     "start a conversation by speaking")
        while self.running:
            try:
                if detecting:
                    if not self.vision.snapshot().present:
                        self.state = "IDLE"
                        self.maybe_lonely()
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
                    self.maybe_lonely()
                    continue

                while self.running and self.converse_turn():
                    if detecting and not self.vision.snapshot().present:
                        log.info("VISION guest left mid-conversation")
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
        if self.preloader is not None:
            self.preloader.stop()
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
