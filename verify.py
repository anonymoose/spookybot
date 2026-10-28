"""Headless verification: stub the hardware and hosted APIs, then exercise the
real logic — VAD state machine, sentence chunking, jaw envelope, servo mapping,
vision box parsing, and a full orchestrator conversation cycle.

This runs on any machine. It does not need a Pi, a camera, or API keys.
"""
from __future__ import annotations

import sys
import types
import threading
import time
import numpy as np

# ---------------------------------------------------------------- stubs ----

class FakeStream:
    def __init__(self, **kw):
        self.kw = kw
        self.callback = kw.get("callback")
        self.started = False
        self._t = None
        self._run = False

    def start(self):
        self.started = True
        self._run = True
        if self.kw.get("_drive_output"):
            self._t = threading.Thread(target=self._pump, daemon=True)
            self._t.start()

    def _pump(self):
        frames = self.kw.get("blocksize", 512)
        while self._run:
            out = np.zeros((frames, 1), dtype=np.int16)
            try:
                self.callback(out, frames, None, None)
            except Exception:
                break
            time.sleep(0.001)

    def stop(self):
        self._run = False
    def close(self):
        self._run = False


class CallbackStop(Exception):
    pass


sd = types.ModuleType("sounddevice")
sd.CallbackStop = CallbackStop
sd.query_devices = lambda: [
    {"name": "ReSpeaker 4 Mic Array (UAC1.0)", "max_input_channels": 6,
     "max_output_channels": 2, "default_samplerate": 16000.0},
    {"name": "UACDemoV1.0", "max_input_channels": 0,
     "max_output_channels": 2, "default_samplerate": 48000.0},
]
sd.default = types.SimpleNamespace(device=(0, 1))

_output_streams = []

def _input_stream(**kw):
    return FakeStream(**kw)

def _output_stream(**kw):
    kw["_drive_output"] = True
    s = FakeStream(**kw)
    _output_streams.append(s)
    return s

sd.InputStream = _input_stream
sd.OutputStream = _output_stream
sys.modules["sounddevice"] = sd

# webrtcvad: mark anything above a modest amplitude as speech
vadmod = types.ModuleType("webrtcvad")
class _Vad:
    def __init__(self, level): self.level = level
    def is_speech(self, data, rate):
        a = np.frombuffer(data, dtype=np.int16)
        return float(np.sqrt(np.mean(a.astype(np.float32) ** 2))) > 200
vadmod.Vad = _Vad
sys.modules["webrtcvad"] = vadmod

# anthropic
anth = types.ModuleType("anthropic")
class _Stream:
    """Stands in for the SDK's stream: yields events like the real one (text
    events, and a content_block_start for a server-side web search)."""
    def __init__(self, script, stop_reason="end_turn"):
        self.script, self.stop_reason = script, stop_reason
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __iter__(self):
        for item in self.script:
            if item in ("<search>", "<code>"):
                block = types.SimpleNamespace(
                    type="server_tool_use",
                    name="web_search" if item == "<search>" else "code_execution")
                yield types.SimpleNamespace(type="content_block_start",
                                            content_block=block)
            else:
                yield types.SimpleNamespace(type="text", text=item)
    def get_final_message(self):
        usage = types.SimpleNamespace(input_tokens=900, output_tokens=40,
                                      cache_creation_input_tokens=0,
                                      cache_read_input_tokens=0)
        return types.SimpleNamespace(usage=usage, stop_reason=self.stop_reason,
                                     content=[types.SimpleNamespace(type="text")])
class _Messages:
    # Tests can queue (script, stop_reason) turns here; each stream() takes one.
    queued = []
    calls = []
    def stream(self, **kw):
        _Messages.last_kwargs = kw
        _Messages.calls.append(kw)
        if _Messages.queued:
            script, stop = _Messages.queued.pop(0)
            return _Stream(script, stop)
        return _Stream(["Well ", "well. ", "A pirate, ", "is it? ",
                        "Bold ", "choice."])
def _create(self, **kw):
    _Messages.create_calls.append(kw)
    text = (_Messages.create_queued.pop(0) if _Messages.create_queued
            else f"Pun number {len(_Messages.create_calls)}. Punchline {len(_Messages.create_calls)}.")
    usage = types.SimpleNamespace(input_tokens=500, output_tokens=30,
                                  cache_creation_input_tokens=0,
                                  cache_read_input_tokens=0)
    return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=text)],
                                 usage=usage, stop_reason="end_turn")
_Messages.create_calls, _Messages.create_queued = [], []
_Messages.create = _create
class _Anthropic:
    def __init__(self, *a, **kw): self.messages = _Messages()
anth.Anthropic = _Anthropic
sys.modules["anthropic"] = anth

# openai (STT)
oai = types.ModuleType("openai")
class _Transcriptions:
    def create(self, **kw):
        return types.SimpleNamespace(text="what are you supposed to be")
class _OpenAI:
    def __init__(self, *a, **kw):
        self.audio = types.SimpleNamespace(transcriptions=_Transcriptions())
oai.OpenAI = _OpenAI
sys.modules["openai"] = oai

# requests (ElevenLabs)
req = types.ModuleType("requests")
class _Resp:
    status_code = 200
    def raise_for_status(self): pass
    def iter_content(self, chunk_size=4096):
        # ~0.6s of 22050 Hz noise, in chunks
        total = np.random.randint(-8000, 8000, 13000).astype(np.int16).tobytes()
        for i in range(0, len(total), chunk_size):
            yield total[i:i + chunk_size]
req.post = lambda *a, **kw: _Resp()
sys.modules["requests"] = req

import os
os.environ.setdefault("ELEVENLABS_API_KEY", "test")

# ------------------------------------------------------------- the tests ----

FAILURES = []

def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def test_config():
    print("\nconfig")
    from spookybot.config import CONFIG
    check("audio block size divides cleanly",
          CONFIG.audio.sample_rate * CONFIG.audio.block_ms % 1000 == 0,
          f"{CONFIG.audio.sample_rate * CONFIG.audio.block_ms // 1000} samples")
    check("VAD block_ms is 10/20/30", CONFIG.audio.block_ms in (10, 20, 30))
    check("jaw min < max", CONFIG.servos.jaw.min_deg < CONFIG.servos.jaw.max_deg)


def test_device_matching():
    print("\ndevice matching")
    from spookybot.audio_io import find_device
    check("finds ReSpeaker as input", find_device("ReSpeaker", True) == 0)
    check("finds UACDemo as output", find_device("UACDemo", False) == 1)
    check("ReSpeaker not matched as output-only name",
          find_device("UACDemo", True) is None)
    check("unknown name falls back to None", find_device("nope", True) is None)


def test_servo_mapping():
    print("\nservo mapping")
    from spookybot.config import CONFIG
    from spookybot.servos import ServoBus
    bus = ServoBus(CONFIG.servos)
    try:
        bus.set_normalized("jaw", 0.0)
        lo = bus._targets["jaw"]
        bus.set_normalized("jaw", 1.0)
        hi = bus._targets["jaw"]
        check("jaw 0.0 -> min_deg", abs(lo - CONFIG.servos.jaw.min_deg) < 0.01)
        check("jaw 1.0 -> max_deg", abs(hi - CONFIG.servos.jaw.max_deg) < 0.01)
        bus.set_normalized("jaw", 5.0)
        check("out-of-range clamps", abs(bus._targets["jaw"] - hi) < 0.01)
        bus.look_at(0.0, 0.5)
        left = bus._targets["pan"]
        bus.look_at(1.0, 0.5)
        right = bus._targets["pan"]
        check("look_at pans across range", left != right,
              f"{left:.0f}° vs {right:.0f}°")
        time.sleep(0.3)
        check("smoothing thread moves toward target",
              bus._current["pan"] != CONFIG.servos.pan.rest_deg,
              f"now {bus._current['pan']:.1f}°")
    finally:
        bus.close()


def test_eye_fade():
    print("\neye fade")
    from spookybot.servos import eye_level
    check("dark at the start of a cycle", eye_level(0.0, 2.0) < 0.001)
    check("full brightness half way through", abs(eye_level(1.0, 2.0) - 1.0) < 0.001)
    check("dark again after one 2s cycle", eye_level(2.0, 2.0) < 0.001)
    samples = [eye_level(i / 100, 2.0) for i in range(201)]
    check("always within 0..1", all(0.0 <= v <= 1.0 for v in samples))
    check("brightens through the first half",
          all(a <= b for a, b in zip(samples[:100], samples[1:101])))


def test_chat_cost():
    print("\nchat cost accounting")
    from spookybot.brain import call_cost
    usage = types.SimpleNamespace(input_tokens=900, output_tokens=40,
                                  cache_creation_input_tokens=0,
                                  cache_read_input_tokens=0)
    check("sonnet-5-5 priced at $2/$10", abs(call_cost("claude-sonnet-5-5", usage) - 0.0022) < 1e-9)
    check("sonnet-5 isn't mistaken for sonnet-5-5 (or vice versa)",
          call_cost("claude-sonnet-5", usage) == call_cost("claude-sonnet-5-5", usage))
    check("dated model IDs match their family",
          abs(call_cost("claude-haiku-4-5-20251001", usage) - 0.0011) < 1e-9)
    check("unknown model has no cost", call_cost("some-other-model", usage) is None)


def test_jaw_envelope():
    print("\njaw envelope shaping")
    from spookybot.audio_io import Speaker
    s = Speaker._shape
    check("silence closes the jaw", s(0.0) == 0.0)
    check("full scale opens fully", s(1.0) >= 0.99, f"{s(1.0):.2f}")
    check("monotonic", all(s(i/20) <= s((i+1)/20) + 1e-9 for i in range(20)))
    check("quiet speech still visible", s(0.15) > 0.3,
          f"0.15 -> {s(0.15):.2f}")
    check("never exceeds 1.0", max(s(i/50) for i in range(51)) <= 1.0)


def test_vad_state_machine():
    print("\nVAD utterance capture")
    from spookybot.config import CONFIG
    from spookybot.vad import UtteranceDetector

    class ScriptedMic:
        """Silence, then speech, then silence — the shape of a real utterance."""
        def __init__(self, block):
            self.block = block
            self.seq = ([0] * 10) + ([1] * 40) + ([0] * 50)
            self.i = 0
        def read(self, timeout=0.5):
            if self.i >= len(self.seq):
                return np.zeros(self.block, dtype=np.int16)
            loud = self.seq[self.i]
            self.i += 1
            amp = 4000 if loud else 30
            return (np.random.randn(self.block) * amp).astype(np.int16)

    det = UtteranceDetector(CONFIG.vad, CONFIG.audio.sample_rate,
                            CONFIG.audio.block_ms)
    block = int(CONFIG.audio.sample_rate * CONFIG.audio.block_ms / 1000)
    fired = []
    audio = det.capture(ScriptedMic(block), timeout_s=5.0,
                        on_speech_start=lambda: fired.append(1))
    check("captured an utterance", audio is not None)
    if audio is not None:
        dur = len(audio) / CONFIG.audio.sample_rate
        check("duration is plausible", 0.8 < dur < 3.0, f"{dur:.2f}s")
        check("preroll included", len(audio) > 40 * block * 0.9)
    check("on_speech_start fired once", len(fired) == 1)

    class SilentMic:
        def read(self, timeout=0.5):
            return (np.random.randn(block) * 20).astype(np.int16)
    t0 = time.monotonic()
    quiet = det.capture(SilentMic(), timeout_s=1.0)
    check("silence times out to None", quiet is None)
    check("timeout is honored", time.monotonic() - t0 < 2.5,
          f"{time.monotonic()-t0:.1f}s")


def test_sentence_chunking():
    print("\nbrain sentence streaming")
    from spookybot.config import CONFIG
    from spookybot.brain import Brain
    brain = Brain(CONFIG.brain)
    sentences = list(brain.respond("hi"))
    check("splits into sentences", len(sentences) >= 2, f"{sentences}")
    check("no empty sentences", all(s.strip() for s in sentences))
    joined = " ".join(sentences)
    check("no text lost", "pirate" in joined.lower() and "Bold" in joined)
    check("history recorded", len(brain.conversation.messages) == 2)
    check("assistant turn stored",
          brain.conversation.messages[-1]["role"] == "assistant")


def test_image_and_trim():
    print("\nbrain vision + context trimming")
    from spookybot.config import BrainConfig
    from spookybot.brain import Brain
    cfg = BrainConfig()
    cfg.history_turns = 2
    brain = Brain(cfg)
    fake_jpeg = b"\xff\xd8\xff\xe0" + b"x" * 900

    list(brain.respond("a guest", image_jpeg=fake_jpeg, scene_note="look"))
    first = brain.conversation.messages[0]["content"]
    check("image block attached", any(b["type"] == "image" for b in first))
    check("scene note prefixed",
          any("look" in b.get("text", "") for b in first if b["type"] == "text"))

    for i in range(4):
        list(brain.respond(f"turn {i}", image_jpeg=fake_jpeg))
    msgs = brain.conversation.messages
    check("history trimmed", len(msgs) <= cfg.history_turns * 2,
          f"{len(msgs)} messages")
    total_images = sum(
        1 for m in msgs if isinstance(m.get("content"), list)
        for b in m["content"] if b.get("type") == "image")
    check("exactly one image retained", total_images == 1,
          f"{total_images} present")
    check("history starts with a user turn (API requires it)",
          msgs[0]["role"] == "user", f"starts with {msgs[0]['role']}")
    check("roles alternate", all(
        msgs[i]["role"] != msgs[i + 1]["role"] for i in range(len(msgs) - 1)))


def test_vision_box_parsing():
    print("\nvision detection logic")
    from spookybot.config import VisionConfig
    from spookybot.vision import Vision

    cfg = VisionConfig()
    cfg.enabled = False
    v = Vision(cfg)
    v.cfg = cfg

    # Hailo NMS layout: per-class lists of [y0, x0, y1, x1, score], normalized.
    # One close person, one distant person, one car (class 2).
    empty = [np.zeros((0, 5)) for _ in range(80)]
    outputs = list(empty)
    outputs[0] = np.array([[0.20, 0.40, 0.90, 0.60, 0.9],
                           [0.45, 0.10, 0.55, 0.15, 0.8]])
    outputs[2] = np.array([[0.10, 0.70, 0.95, 0.99, 0.9]])

    for _ in range(cfg.arrive_frames):
        v._process(outputs)
    g = v.snapshot()
    check("person detected as present", g.present)
    check("distant person filtered out", g.count == 1, f"count={g.count}")
    check("car ignored", g.count == 1)
    check("center_x is box midpoint", abs(g.center_x - 0.50) < 0.01,
          f"{g.center_x:.2f}")
    check("box_height correct", abs(g.box_height - 0.70) < 0.01,
          f"{g.box_height:.2f}")

    for _ in range(cfg.depart_frames):
        v._process(empty)
    check("departure detected", not v.snapshot().present)

    # hysteresis: a single dropped frame must not clear presence
    for _ in range(cfg.arrive_frames):
        v._process(outputs)
    v._process(empty)
    check("one missed frame doesn't clear presence", v.snapshot().present)


def test_speaker_playback():
    print("\nspeaker + jaw integration")
    from spookybot.config import CONFIG
    from spookybot.audio_io import Speaker
    jaw_values = []
    sp = Speaker(CONFIG.audio, jaw_callback=jaw_values.append)
    sp.start()
    loud = (np.random.randn(8000) * 9000).astype(np.int16).tobytes()
    sp.feed(loud)
    sp.done_feeding()
    ok = sp.wait(timeout=5.0)
    sp.close()
    check("playback completes", ok)
    check("jaw was driven", len(jaw_values) > 5, f"{len(jaw_values)} updates")
    check("jaw opened at some point", max(jaw_values or [0]) > 0.4,
          f"peak {max(jaw_values or [0]):.2f}")
    check("jaw values in range", all(0.0 <= v <= 1.0 for v in jaw_values))


def test_orchestrator():
    print("\norchestrator conversation cycle")
    from spookybot.config import CONFIG
    CONFIG.vision.enabled = False
    CONFIG.listen_timeout_s = 1.0
    from spookybot.main import SpookyBot

    bot = SpookyBot(CONFIG)
    try:
        spoken = []
        real_say = bot.say
        bot.say = lambda s: (spoken.extend(list(s)), real_say(iter([])))[0]

        bot.greet()
        check("greeting produced", len(spoken) > 0, f"{spoken[:1]}")

        # Feed a scripted utterance through the real detector
        block = bot.mic.block
        frames = ([np.zeros(block, dtype=np.int16)] * 3 +
                  [(np.random.randn(block) * 4000).astype(np.int16)] * 30 +
                  [np.zeros(block, dtype=np.int16)] * 45)
        for f in frames:
            try:
                bot.mic._q.put_nowait(f)
            except Exception:
                pass

        spoken.clear()
        cont = bot.converse_turn()
        check("turn completed", cont)
        check("bot replied to guest", len(spoken) > 0, f"{spoken[:1]}")

        spoken.clear()
        bot.mic.flush()
        check("silence ends the conversation", not bot.converse_turn())
    finally:
        bot.shutdown()


def test_vision_degrades():
    print("\nvision degrades instead of crashing")
    import spookybot.vision as vmod
    from spookybot.config import VisionConfig

    class FakePicam:
        def __init__(self):
            self.streams = None
        def create_preview_configuration(self, **streams):
            self.streams = streams
            return streams
        def configure(self, cfg): pass
        def start(self): pass
        def stop(self): pass
        def close(self): pass

    class NoHailo:
        def __init__(self, model):
            raise RuntimeError("HAILO_DRIVER_NOT_INSTALLED")

    class NoCamera:
        def __init__(self):
            raise RuntimeError("no cameras available")

    saved = (vmod._HAVE_CAMERA, getattr(vmod, "Hailo", None),
             getattr(vmod, "Picamera2", None), vmod.time.sleep)
    vmod._HAVE_CAMERA = True
    vmod.time.sleep = lambda s: None
    try:
        cfg = VisionConfig()
        cfg.enabled = True

        vmod.Hailo, vmod.Picamera2 = NoHailo, FakePicam
        v = vmod.Vision(cfg)
        check("missing Hailo keeps the camera", v._picam is not None)
        check("missing Hailo means not detecting", not v.detecting)
        check("camera-only asks for no detector stream",
              v._picam.streams is not None and "lores" not in v._picam.streams)
        v.close()

        vmod.Picamera2 = NoCamera
        v = vmod.Vision(cfg)
        check("missing camera disables vision", v._picam is None and not v.detecting)
    finally:
        (vmod._HAVE_CAMERA, vmod.Hailo, vmod.Picamera2, vmod.time.sleep) = saved


def test_voice_arrival():
    print("\nvoice-activated arrival (no presence detection)")
    from spookybot.config import CONFIG
    CONFIG.vision.enabled = False
    CONFIG.listen_timeout_s = 1.0
    CONFIG.wake_phrase = ""
    from spookybot.main import SpookyBot

    bot = SpookyBot(CONFIG)
    try:
        check("vision reports no detection", not bot.vision.detecting)
        spoken, calls = [], []
        real_say, real_respond = bot.say, bot.brain.respond
        bot.say = lambda s: (spoken.extend(list(s)), real_say(iter([])))[0]
        bot.brain.respond = lambda text, image_jpeg=None, scene_note=None: (
            calls.append((text, image_jpeg, scene_note)),
            real_respond(text, image_jpeg=image_jpeg, scene_note=scene_note))[1]
        bot.vision.capture_jpeg = lambda: b"\xff\xd8fake-jpeg"

        bot.mic.flush()
        check("silence starts nothing", not bot.start_by_voice())
        check("no greeting to an empty room", spoken == [], f"{spoken[:1]}")

        block = bot.mic.block
        frames = ([np.zeros(block, dtype=np.int16)] * 3 +
                  [(np.random.randn(block) * 4000).astype(np.int16)] * 30 +
                  [np.zeros(block, dtype=np.int16)] * 45)
        for f in frames:
            try:
                bot.mic._q.put_nowait(f)
            except Exception:
                pass
        check("speaking starts a conversation", bot.start_by_voice())
        check("guest got a reply", len(spoken) > 0, f"{spoken[:1]}")
        check("first reply carries the costume still",
              bool(calls) and calls[0][1] == b"\xff\xd8fake-jpeg")
    finally:
        bot.shutdown()


def test_wake_phrase():
    print("\nwake phrase (\"hello bones\")")
    from spookybot.wake import strip_wake_phrase as w
    check("plain match", w("Hello Bones", "hello bones") == "")
    check("punctuation and trailing words", w("Hello, Bones! Nice head.", "hello bones") == "Nice head.")
    check("matches mid-sentence", w("okay so hello bones what are you", "hello bones") == "what are you")
    check("common mishearing", w("Hullo bone, who are you?", "hello bones") == "who are you?")
    check("no phrase -> None", w("what are you supposed to be", "hello bones") is None)
    check("half a phrase -> None", w("hello there", "hello bones") is None)
    check("'bones' alone -> None", w("my bones hurt", "hello bones") is None)

    from spookybot.config import CONFIG
    CONFIG.vision.enabled = False
    CONFIG.listen_timeout_s = 1.0
    CONFIG.wake_phrase = "hello bones"
    from spookybot.main import SpookyBot

    bot = SpookyBot(CONFIG)
    try:
        spoken, heard = [], []
        real_say, real_respond = bot.say, bot.brain.respond
        bot.say = lambda s: (spoken.extend(list(s)), real_say(iter([])))[0]
        prompts = []
        bot.brain.respond = lambda text, image_jpeg=None, scene_note=None: (
            heard.append(text), prompts.append(scene_note or ""), real_respond(text, image_jpeg=image_jpeg,
                                             scene_note=scene_note))[2]

        def speak(phrase):
            bot.stt.transcribe = lambda audio: phrase
            block = bot.mic.block
            frames = ([np.zeros(block, dtype=np.int16)] * 3 +
                      [(np.random.randn(block) * 4000).astype(np.int16)] * 30 +
                      [np.zeros(block, dtype=np.int16)] * 45)
            for f in frames:
                try:
                    bot.mic._q.put_nowait(f)
                except Exception:
                    pass
            return bot.start_by_voice()

        bot.mic.flush()
        check("speech without the phrase is ignored",
              not speak("what are you supposed to be"))
        check("and it stays silent", spoken == [] and heard == [], f"{spoken[:1]}")

        check("old phrase no longer wakes it",
              not speak("Hey Bones, who are you?"))

        bot.mic.flush()
        calls = []
        real_turn = bot.converse_turn
        bot.converse_turn = lambda **kw: (calls.append(kw), False)[1]
        check("phrase starts a conversation", speak("Hello Bones, who are you?"))
        check("first thing said is the intro, not an answer",
              len(heard) == 1 and "says hello" in heard[0], f"{heard}")
        check("intro asked for ~30 words and a fresh wording",
              "30 words" in prompts[0] and "differently" in prompts[0])
        check("intro spoken", len(spoken) > 0)
        check("then it listens for the guest", len(calls) == 1)

        bot.mic.flush()
        speak("Hello Bones.")
        check("second intro is told not to reuse the first",
              "Do not reuse" in prompts[1], f"{prompts[1][-80:]}")
        bot.converse_turn = real_turn
    finally:
        bot.shutdown()


def test_still_every_turn():
    print("\nfresh camera still on every turn")
    from spookybot.config import CONFIG
    CONFIG.vision.enabled = False
    CONFIG.listen_timeout_s = 1.0
    CONFIG.wake_phrase = ""
    from spookybot.main import SpookyBot

    bot = SpookyBot(CONFIG)
    try:
        images = []
        real_respond = bot.brain.respond
        bot.say = lambda s: list(s)
        bot.brain.respond = lambda text, image_jpeg=None, scene_note=None: (
            images.append(image_jpeg),
            real_respond(text, image_jpeg=image_jpeg, scene_note=scene_note))[1]
        stills = iter([b"\xff\xd8one", b"\xff\xd8two"])
        bot.vision.capture_jpeg = lambda: next(stills)
        audio = np.zeros(1600, dtype=np.int16)
        bot.converse_turn(audio=audio)
        bot.converse_turn(audio=audio)
        check("every turn gets a still, not just the first",
              images == [b"\xff\xd8one", b"\xff\xd8two"], f"{images}")
        bot.converse_turn(audio=audio, image=b"\xff\xd8given")
        check("a still passed in is used as is", images[-1] == b"\xff\xd8given")
    finally:
        bot.shutdown()


def test_startup_announcement():
    print("\nstartup announcement")
    from datetime import datetime
    from spookybot.main import spoken_time
    check("afternoon", spoken_time(datetime(2026, 10, 27, 19, 28)) == "7:28 PM")
    check("morning, minutes padded", spoken_time(datetime(2026, 10, 27, 8, 5)) == "8:05 AM")
    check("midnight is 12 AM", spoken_time(datetime(2026, 10, 27, 0, 0)) == "12:00 AM")
    check("noon is 12 PM", spoken_time(datetime(2026, 10, 27, 12, 30)) == "12:30 PM")

    from spookybot.config import CONFIG
    CONFIG.vision.enabled = False
    from spookybot.main import SpookyBot
    bot = SpookyBot(CONFIG)
    try:
        spoken = []
        bot.say = lambda s: spoken.extend(list(s))
        bot.announce_ready()
        line = spoken[0] if spoken else ""
        check("announces name and time",
              line.startswith("Hello, my name is Bones. It is ")
              and line.endswith(" Eastern time."), f"{line!r}")
        bot.say = lambda s: (_ for _ in ()).throw(RuntimeError("speaker gone"))
        bot.announce_ready()
        check("a failed announcement doesn't stop startup", True)
    finally:
        bot.shutdown()


def test_eye_modes():
    print("\neye modes follow what the bot is doing")
    from spookybot.config import CONFIG
    from spookybot.servos import ServoBus, eye_level
    CONFIG.vision.enabled = False
    bus = ServoBus(CONFIG.servos)
    try:
        t = bus._eyes_t0 + 0.7
        check("default is the slow pulse",
              bus.eye_mode == "pulse"
              and abs(bus.eye_brightness(t) - eye_level(0.7, CONFIG.servos.eyes_period_s)) < 1e-9)
        bus.set_eye_mode("steady")
        check("steady is full on, always",
              all(bus.eye_brightness(t + i * 0.37) == 1.0 for i in range(20)))
        bus.set_eye_mode("flicker")
        levels = [bus.eye_brightness(t + i * 0.2) for i in range(60)]
        check("flicker stays in range", all(0.0 < v <= 1.0 for v in levels))
        check("flicker actually varies", len(set(levels)) > 10, f"{len(set(levels))} levels")
        check("flicker holds a level between changes",
              bus.eye_brightness(bus._flicker_until - 0.001)
              == bus.eye_brightness(bus._flicker_until - 0.002))
        try:
            bus.set_eye_mode("strobe")
            check("unknown mode rejected", False)
        except ValueError:
            check("unknown mode rejected", True)
    finally:
        bus.close()

    from spookybot.main import SpookyBot
    bot = SpookyBot(CONFIG)
    try:
        modes = {}
        for st in ("IDLE", "LISTEN", "THINK", "GREET", "SPEAK"):
            bot.state = st
            modes[st] = bot.servos.eye_mode
        check("pulse while waiting for a guest",
              modes["IDLE"] == modes["LISTEN"] == "pulse", f"{modes}")
        check("flicker while waiting on Claude",
              modes["THINK"] == modes["GREET"] == modes["SPEAK"] == "flicker", f"{modes}")
        check("state still reads back", bot.state == "SPEAK")

        seen = []
        real_set = bot.servos.set_eye_mode
        bot.servos.set_eye_mode = lambda m: (seen.append(m), real_set(m))[1]
        bot.tts.stream = lambda sentence: iter([b"\x00\x01" * 400])
        bot.say(iter(["Hello there."]))
        check("steady once the voice starts, then back to pulse",
              seen[-2:] == ["steady", "pulse"] and bot.servos.eye_mode == "pulse", f"{seen}")
    finally:
        bot.shutdown()


def test_quiet_phrase():
    print("\nquiet phrase (\"quiet bones\")")
    from spookybot.wake import strip_wake_phrase as w
    check("matches", w("Quiet Bones!", "quiet bones") == "")
    check("matches a common mishearing", w("quite bones, enough", "quiet bones") == "enough")
    check("doesn't match 'quiet' alone", w("it's quiet in here", "quiet bones") is None)

    from spookybot.config import CONFIG
    CONFIG.vision.enabled = False
    CONFIG.listen_timeout_s = 1.0
    CONFIG.quiet_phrase = "quiet bones"
    from spookybot.main import SpookyBot
    bot = SpookyBot(CONFIG)
    try:
        spoken, heard = [], []
        real_say, real_respond = bot.say, bot.brain.respond
        bot.say = lambda s: (spoken.extend(list(s)), real_say(iter([])))[0]
        bot.brain.respond = lambda text, image_jpeg=None, scene_note=None: (
            heard.append(text), real_respond(text, image_jpeg=image_jpeg,
                                             scene_note=scene_note))[1]
        audio = np.zeros(1600, dtype=np.int16)

        bot.stt.transcribe = lambda a: "Quiet Bones, you talk too much"
        check("quiet phrase ends the conversation", bot.converse_turn(audio=audio) is False)
        check("he says As you wish", spoken == ["As you wish."], f"{spoken}")
        check("and does not ask Claude anything", heard == [])

        spoken.clear()
        bot.stt.transcribe = lambda a: "what are you supposed to be"
        check("normal speech carries on", bot.converse_turn(audio=audio) is True)
        check("and is answered", len(heard) == 1 and len(spoken) > 0)

        CONFIG.quiet_phrase = ""
        spoken.clear(); heard.clear()
        bot.stt.transcribe = lambda a: "quiet bones"
        check("empty setting turns it off", bot.converse_turn(audio=audio) is True
              and heard == ["quiet bones"])
    finally:
        CONFIG.quiet_phrase = "quiet bones"
        bot.shutdown()


def test_liveview():
    print("\nlive camera view")
    try:
        from PIL import Image
    except ImportError:
        print("  [SKIP] Pillow not installed here (the Pi has it via picamera2)")
        return
    import io
    import urllib.request
    from spookybot import liveview

    img = Image.new("RGB", (640, 360), (40, 40, 40))
    before = img.tobytes()
    liveview.draw_overlay(img, [(0.1, 0.2, 0.9, 0.5, 0.91, ""),
                                (0.2, 0.7, 0.4, 0.8, 0.31, "low")], "GUEST x1")
    px = img.load()
    check("accepted box drawn in green", px[int(0.2 * 640), int(0.5 * 360)] == liveview.GREEN)
    check("rejected box drawn in orange", px[int(0.7 * 640), int(0.3 * 360)] == liveview.ORANGE)
    corner = [px[x, y] for x in range(0, 70) for y in range(0, 24)]
    check("header text drawn on a dark strip",
          img.tobytes() != before and (0, 0, 0) in corner and (255, 255, 255) in corner)

    from spookybot.config import CONFIG
    CONFIG.vision.enabled = False
    from spookybot.vision import Vision
    v = Vision(CONFIG.vision)
    check("no camera -> no frame, no crash", v.render_overlay() is None)

    buf = io.BytesIO(); img.save(buf, format="JPEG")
    frames = [buf.getvalue()]
    server = liveview.start(lambda: frames[0], 0)
    try:
        port = server.server_address[1]
        check("listens on localhost only", server.server_address[0] == "127.0.0.1",
              f"{server.server_address[0]}")
        r = urllib.request.urlopen(f"http://127.0.0.1:{port}/snapshot.jpg", timeout=5)
        body = r.read()
        check("snapshot is a jpeg", r.status == 200
              and r.headers["Content-Type"] == "image/jpeg" and body[:2] == b"\xff\xd8")
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).read()
        check("page embeds the stream", b"/stream" in page)
        r = urllib.request.urlopen(f"http://127.0.0.1:{port}/stream", timeout=5)
        chunk = r.read(4096)
        check("stream is multipart mjpeg",
              "multipart/x-mixed-replace" in r.headers["Content-Type"]
              and b"--frame" in chunk and b"image/jpeg" in chunk)
        r.close()
        import json
        import logging
        logging.getLogger("spookybot.test").warning("VOICE hello from the log panel")
        j = json.loads(urllib.request.urlopen(
            f"http://127.0.0.1:{port}/logs?after=0", timeout=5).read())
        check("log panel gets the bot's log lines",
              any("hello from the log panel" in l["text"] for l in j["lines"]), f"{j['last']}")
        j2 = json.loads(urllib.request.urlopen(
            f"http://127.0.0.1:{port}/logs?after={j['last']}", timeout=5).read())
        check("and only the new ones on the next poll", j2["lines"] == [])
        check("page has the log panel", b'id=log' in page and b"/logs?after=" in page)
        frames[0] = None
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/snapshot.jpg", timeout=5)
            check("503 when the camera has no frame", False)
        except urllib.error.HTTPError as exc:
            check("503 when the camera has no frame", exc.code == 503)
    finally:
        server.shutdown()
        logging.getLogger().removeHandler(server.log_buffer)


def test_web_search():
    print("\nweb search (Anthropic server-side tool)")
    from spookybot.config import BrainConfig
    from spookybot.brain import Brain
    M = sys.modules["anthropic"].Anthropic().messages.__class__

    cfg = BrainConfig()
    check("on by default", cfg.web_search is True)
    brain = Brain(cfg)
    M.calls.clear()
    list(brain.respond("who won the game?"))
    kw = M.calls[-1]
    tool = (kw.get("tools") or [{}])[0]
    check("web_search tool declared on the request",
          tool.get("type") == "web_search_20250305" and tool.get("name") == "web_search",
          f"{tool}")
    check("limited to one search per reply", tool.get("max_uses") == 1)
    check("reply budget raised so queries fit", kw["max_tokens"] >= 400,
          f"{kw['max_tokens']}")
    check("no blocked domains unless configured", "blocked_domains" not in tool)

    cfg2 = BrainConfig()
    cfg2.web_blocked_domains = "bad.example, worse.example ,"
    t2 = Brain(cfg2)._tools()[0]
    check("blocked domains passed through, trimmed",
          t2.get("blocked_domains") == ["bad.example", "worse.example"], f"{t2}")

    off = BrainConfig()
    off.web_search = False
    M.calls.clear()
    list(Brain(off).respond("hi"))
    check("off -> no tools at all and the old token budget",
          "tools" not in M.calls[-1] and M.calls[-1]["max_tokens"] == off.max_tokens)

    # A search mid-answer: he speaks first, goes quiet, then answers.
    waits = []
    brain = Brain(BrainConfig())
    brain.on_wait = waits.append
    M.queued.append((["Let me check.", "<search>", "The Heat won, ", "110 to 98."], "end_turn"))
    said = list(brain.respond("who won the game?"))
    check("sentences before and after the search are all spoken",
          said == ["Let me check.", "The Heat won, 110 to 98."], f"{said}")
    check("wait signalled on search start and cleared when talking resumes",
          waits == [True, False], f"{waits}")
    check("history keeps only the spoken text",
          brain.conversation.messages[-1]["content"] == "Let me check. The Heat won, 110 to 98.")
    check("search counted", brain.searches == 1)

    # hidden code-execution steps are not searches and must not stall the eyes
    waits.clear()
    brain = Brain(BrainConfig())
    brain.on_wait = waits.append
    M.queued.append((["<code>", "Done."], "end_turn"))
    list(brain.respond("x"))
    check("a code-execution step is not counted as a search",
          brain.searches == 0 and waits == [], f"{brain.searches} {waits}")

    # pause_turn: the reply is resumed instead of cut off
    M.calls.clear()
    brain = Brain(BrainConfig())
    M.queued.append((["Checking. ", "<search>"], "pause_turn"))
    M.queued.append((["It is sunny."], "end_turn"))
    said = list(brain.respond("weather?"))
    check("pause_turn is resumed", said == ["Checking.", "It is sunny."], f"{said}")
    check("resume resends the partial answer",
          len(M.calls) == 2 and M.calls[1]["messages"][-1]["role"] == "assistant")

    # never loops forever
    M.calls.clear()
    brain = Brain(BrainConfig())
    for _ in range(6):
        M.queued.append((["Hmm. "], "pause_turn"))
    list(brain.respond("endless"))
    check("gives up after a few resumes", len(M.calls) == 3, f"{len(M.calls)} calls")
    M.queued.clear()

    # eyes: flicker during the search wait, steady again once speaking
    from spookybot.config import CONFIG
    CONFIG.vision.enabled = False
    from spookybot.main import SpookyBot
    bot = SpookyBot(CONFIG)
    try:
        bot._voice_started = True
        bot.brain.on_wait(True)
        check("eyes flicker while searching", bot.servos.eye_mode == "flicker")
        bot.brain.on_wait(False)
        check("eyes steady once he talks again", bot.servos.eye_mode == "steady")
        bot._voice_started = False
        bot.servos.set_eye_mode("flicker")
        bot.brain.on_wait(False)
        check("but not before any voice has started", bot.servos.eye_mode == "flicker")
    finally:
        bot.shutdown()


def test_preload():
    print("\npreloaded replies (hello bones intros and jokes)")
    from spookybot.config import CONFIG, BrainConfig
    from spookybot.brain import Brain, split_sentences
    from spookybot.preload import Preloader
    from spookybot.tts import Speech
    import spookybot.main as main_mod
    M = sys.modules["anthropic"].Anthropic().messages.__class__

    check("splits a reply into sentences",
          split_sentences("Why? Because. Ha!") == ["Why?", "Because.", "Ha!"])
    check("joke trigger: the word joke, any case, plural",
          all(main_mod.JOKE_RE.search(t) for t in
              ["tell me a joke", "Got any JOKES?", "a joke about cats please"]))
    notes = {main_mod.joke_request([])[1] for _ in range(40)}
    check("each joke is given a theme, and they vary", len(notes) > 3, f"{len(notes)} variants")
    check("jokes are told what to avoid",
          "Do not reuse these jokes: Old joke." in main_mod.joke_request(["Old joke."])[1])
    check("joke trigger ignores other words",
          not any(main_mod.JOKE_RE.search(t) for t in
                  ["what is your name", "jokester", "who are you"]))

    # -- speech cache -------------------------------------------------------
    tts = Speech(CONFIG.tts, CONFIG.audio.playback_rate)
    backend_calls = []
    real_primary = tts._primary
    tts._primary = lambda text: (backend_calls.append(text), real_primary(text))[1]
    check("warm synthesizes and stores", tts.warm("Boo.") and tts.is_cached("Boo."))
    n = len(backend_calls)
    audio = b"".join(tts.stream("Boo."))
    check("a cached sentence plays without calling the speech backend",
          len(audio) > 1000 and len(backend_calls) == n)
    check("and is used once", not tts.is_cached("Boo."))
    b"".join(tts.stream("Boo."))
    check("then it is synthesized live again", len(backend_calls) == n + 1)

    def boom(text):
        raise RuntimeError("speech server down")
        yield b""
    tts._primary = boom
    check("a failed warm reports False and caches nothing (no fallback voice)",
          not tts.warm("Fail.") and not tts.is_cached("Fail."))
    tts._primary = real_primary

    # -- brain.generate / inject_turn ----------------------------------------
    brain = Brain(BrainConfig())
    M.create_calls.clear()
    text = brain.generate("Tell me a joke.", "be punny")
    call = M.create_calls[-1]
    check("generate returns the text", text and text.startswith("Pun number"))
    check("generate sends the note, the persona, and no tools",
          call["messages"][0]["content"].startswith("[be punny]")
          and call["system"] == brain.persona and "tools" not in call)
    check("generate leaves the conversation alone", brain.conversation.messages == [])
    brain.inject_turn("tell me a joke", "Why? Because.")
    check("inject_turn records both sides",
          [m["role"] for m in brain.conversation.messages] == ["user", "assistant"])

    # -- the preloader ------------------------------------------------------
    brain = Brain(BrainConfig())
    tts = Speech(CONFIG.tts, CONFIG.audio.playback_rate)
    seen_avoid = []
    prompts = {"joke": lambda avoid: (seen_avoid.append(list(avoid)), ("Tell me a joke.", "joke"))[1],
               "intro": lambda avoid: ("hello", "intro")}
    pre = Preloader(brain, tts, {"intro": 2, "joke": 3}, prompts, backoff_s=0.05)
    M.create_calls.clear()
    made = pre.fill_once()
    check("fills every queue to its target", pre.counts() == {"intro": 2, "joke": 3}
          and made == 5, f"{pre.counts()} made={made}")
    first = pre.take("joke")
    check("take returns the sentences, and the speech is already cached",
          first and all(tts.is_cached(x) for x in first), f"{first}")
    check("taking one leaves the others", pre.counts()["joke"] == 2)
    pre.fill_once()
    check("refill tops back up to the target", pre.counts()["joke"] == 3)
    check("later jokes are told what to avoid (queued and already used)",
          any(" ".join(first) in a for a in seen_avoid[-1:]) and len(seen_avoid[-1]) >= 3,
          f"{seen_avoid[-1]}")
    check("empty or unknown queue gives None", pre.take("nope") is None)

    busy = {"v": False}
    pre2 = Preloader(Brain(BrainConfig()), tts, {"joke": 2}, prompts,
                     idle=lambda: not busy["v"], backoff_s=0.05)
    busy["v"] = True
    check("does nothing while the bot is busy", pre2.fill_once() == 0 and pre2.counts() == {"joke": 0})
    busy["v"] = False

    flaky = Brain(BrainConfig())
    flaky.generate = lambda *a, **k: None
    pre3 = Preloader(flaky, tts, {"joke": 2}, prompts, backoff_s=0.05)
    check("a failed text call stops the pass without raising",
          pre3.fill_once() == 0 and pre3.counts() == {"joke": 0})
    pre4 = Preloader(Brain(BrainConfig()), tts, {"joke": 1}, prompts, backoff_s=0.05)
    tts._primary = boom
    check("a failed speech call drops the reply instead of queueing silence",
          pre4.fill_once() == 0 and pre4.counts() == {"joke": 0})
    tts._primary = real_primary

    # background thread
    pre5 = Preloader(Brain(BrainConfig()), tts, {"joke": 2}, prompts, backoff_s=0.05)
    pre5.start()
    for _ in range(100):
        if pre5.counts()["joke"] == 2:
            break
        time.sleep(0.05)
    check("the background thread stocks the queue", pre5.counts()["joke"] == 2)
    pre5.take("joke")
    for _ in range(100):
        if pre5.counts()["joke"] == 2:
            break
        time.sleep(0.05)
    check("and refills after a reply is used", pre5.counts()["joke"] == 2)
    pre5.stop()


def test_preload_in_the_bot():
    print("\npreloaded replies used by the bot")
    from spookybot.config import CONFIG
    CONFIG.vision.enabled = False
    CONFIG.wake_phrase = "hello bones"
    CONFIG.quiet_phrase = "quiet bones"
    from spookybot.main import SpookyBot

    bot = SpookyBot(CONFIG)
    try:
        check("a preloader exists by default", bot.preloader is not None)
        spoken, live = [], []
        real_respond = bot.brain.respond
        bot.say = lambda s: spoken.extend(list(s))
        bot.brain.respond = lambda text, image_jpeg=None, scene_note=None: (
            live.append(text), real_respond(text, image_jpeg=image_jpeg,
                                            scene_note=scene_note))[1]
        bot.vision.capture_jpeg = lambda: b"\xff\xd8still"
        bot.converse_turn = lambda **kw: False        # skip the listen after the intro
        audio = np.zeros(1600, dtype=np.int16)

        # intro: queue empty -> live path, exactly as before
        bot.introduce()
        check("empty queue -> the intro is written live", len(live) == 1 and spoken)

        # intro: from the queue
        bot.preloader._queues["intro"].append(["Prepared intro one.", "Prepared intro two."])
        live.clear(); spoken.clear()
        bot.introduce()
        check("queued intro is spoken with no call to Claude",
              spoken == ["Prepared intro one.", "Prepared intro two."] and live == [], f"{spoken}")
        roles = [m["role"] for m in bot.brain.conversation.messages]
        check("and goes into the history so Claude knows what it said",
              roles == ["user", "assistant"]
              and "Prepared intro one." in bot.brain.conversation.messages[-1]["content"])
        check("and into the do-not-repeat memory",
              any("Prepared intro one." in g for g in bot.brain.recent_greetings))

        # jokes
        real_turn = SpookyBot.converse_turn.__get__(bot)
        bot.converse_turn = real_turn
        bot.brain.reset()
        bot.preloader._queues["joke"].append(["Why no skeleton fights?", "No guts."])
        bot.stt.transcribe = lambda a: "Tell me a joke about cats"
        live.clear(); spoken.clear()
        check("a joke request keeps the conversation going",
              bot.converse_turn(audio=audio) is True)
        check("it plays the prepared joke, no Claude call, no camera still",
              spoken == ["Why no skeleton fights?", "No guts."] and live == [], f"{spoken} {live}")
        check("the request and joke are in the history",
              bot.brain.conversation.messages[-2]["content"][0]["text"] == "Tell me a joke about cats"
              and "No guts." in bot.brain.conversation.messages[-1]["content"])

        live.clear(); spoken.clear()
        check("queue empty -> a joke request is answered live",
              bot.converse_turn(audio=audio) is True and len(live) == 1)

        bot.preloader._queues["joke"].append(["Unused joke."])
        bot.stt.transcribe = lambda a: "what is your name"
        live.clear(); spoken.clear()
        bot.converse_turn(audio=audio)
        check("other requests never touch the joke queue",
              len(live) == 1 and bot.preloader.counts()["joke"] == 1)

        CONFIG.preload_intros = CONFIG.preload_jokes = 0
        off = SpookyBot(CONFIG)
        check("both counts 0 -> no preloader at all", off.preloader is None)
        off.shutdown()
    finally:
        CONFIG.preload_intros, CONFIG.preload_jokes = 5, 5
        bot.shutdown()


def test_startup_sequence():
    print("\nstartup announcements around the preload")
    from spookybot.config import CONFIG
    CONFIG.vision.enabled = False
    CONFIG.announce_startup = True
    CONFIG.preload_intros, CONFIG.preload_jokes = 2, 2
    from spookybot.main import SpookyBot

    def run(prep=None):
        bot = SpookyBot(CONFIG)
        events = []
        bot.say = lambda s: events.append(("say", " ".join(list(s))))
        real_fill = bot.preloader.fill_initial
        bot.preloader.fill_initial = lambda *a, **k: (events.append(("fill", None)),
                                                      (prep or real_fill)(*a, **k))[1]
        bot.preloader.start = lambda: events.append(("start", None))
        return bot, events

    bot, events = run()
    try:
        bot.startup_sequence()
        kinds = [e[0] if e[0] != "say" else e[1][:14] for e in events]
        check("order: hello, preloading, fill, ready, then the background refiller",
              kinds == ["Hello, my name", "Preloading int", "fill", "Bones is ready", "start"],
              f"{kinds}")
        check("the lines are exactly as asked",
              [e[1] for e in events if e[0] == "say"][1:] ==
              ["Preloading interactions.", "Bones is ready."])
        check("both queues are full by the time he says ready",
              bot.preloader.counts() == {"intro": 2, "joke": 2}, f"{bot.preloader.counts()}")
        check("eyes pulse again afterwards", bot.servos.eye_mode == "pulse")
    finally:
        bot.shutdown()

    bot, events = run(prep=lambda *a, **k: False)     # stocking fails
    try:
        bot.startup_sequence()
        said = [e[1] for e in events if e[0] == "say"]
        check("if stocking fails he still says ready (replies fall back to live)",
              said[-1] == "Bones is ready." and ("start", None) in events, f"{said}")
    finally:
        bot.shutdown()

    CONFIG.announce_startup = False
    bot, events = run()
    try:
        bot.startup_sequence()
        check("announcements off: silent, no blocking fill, refiller starts",
              events == [("start", None)], f"{events}")
    finally:
        CONFIG.announce_startup = True
        CONFIG.preload_intros, CONFIG.preload_jokes = 5, 5
        bot.shutdown()


def test_mic_gating():
    print("\nhalf-duplex mic gating")
    from spookybot.config import CONFIG
    from spookybot.audio_io import MicStream
    mic = MicStream(CONFIG.audio)
    try:
        loud = (np.random.randn(mic.block) * 5000).astype(np.int16)
        indata = loud.reshape(-1, 1)
        mic._callback(indata, mic.block, None, None)
        check("ungated audio reaches queue", mic.read(timeout=0.2) is not None)

        mic.gate(5.0)
        for _ in range(3):
            mic._callback(indata, mic.block, None, None)
        check("gated audio is dropped", mic.read(timeout=0.2) is None)

        mic.ungate()
        mic._callback(indata, mic.block, None, None)
        check("ungate restores capture", mic.read(timeout=0.2) is not None)

        for _ in range(5):
            mic._callback(indata, mic.block, None, None)
        mic.flush()
        check("flush empties the queue", mic.read(timeout=0.2) is None)
    finally:
        mic.close()


def main():
    print("=" * 60)
    print("spookybot headless verification")
    print("=" * 60)
    for fn in (test_config, test_device_matching, test_servo_mapping,
               test_eye_fade, test_chat_cost, test_jaw_envelope, test_vad_state_machine,
               test_sentence_chunking, test_image_and_trim,
               test_vision_box_parsing, test_speaker_playback,
               test_vision_degrades, test_wake_phrase, test_still_every_turn, test_startup_announcement, test_eye_modes, test_quiet_phrase, test_liveview, test_web_search, test_preload, test_preload_in_the_bot, test_startup_sequence, test_mic_gating,
               test_orchestrator, test_voice_arrival):
        try:
            fn()
        except Exception as exc:
            import traceback
            print(f"  [FAIL] {fn.__name__} raised: {exc}")
            traceback.print_exc()
            FAILURES.append(fn.__name__)

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"FAILURES ({len(FAILURES)}): {FAILURES}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
