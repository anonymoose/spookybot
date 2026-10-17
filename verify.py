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
    def __init__(self, chunks): self.chunks = chunks
    def __enter__(self): return self
    def __exit__(self, *a): return False
    @property
    def text_stream(self):
        for c in self.chunks:
            yield c
class _Messages:
    def stream(self, **kw):
        _Messages.last_kwargs = kw
        return _Stream(["Well ", "well. ", "A pirate, ", "is it? ",
                        "Bold ", "choice."])
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
               test_eye_fade, test_jaw_envelope, test_vad_state_machine,
               test_sentence_chunking, test_image_and_trim,
               test_vision_box_parsing, test_speaker_playback,
               test_vision_degrades, test_mic_gating,
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
