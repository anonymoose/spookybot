"""Central configuration. Everything tunable lives here or in env vars.

Copy .env.example to .env and edit, or just export the vars.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _envf(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


def _envi(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def _envb(key: str, default: bool) -> bool:
    return os.environ.get(key, str(default)).lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------- audio ----

@dataclass
class AudioConfig:
    # Device names are matched as substrings against sounddevice's device list.
    # Run `python -m spookybot.tools.list_devices` to see what your Pi sees.
    input_device: str = _env("SB_INPUT_DEVICE", "ReSpeaker")
    output_device: str = _env("SB_OUTPUT_DEVICE", "ReSpeaker")

    # Capture side. 16k mono is what every STT engine wants.
    sample_rate: int = 16000
    channels: int = 1
    block_ms: int = 20                      # VAD requires 10/20/30ms frames

    # Playback side. TTS returns PCM at this rate.
    playback_rate: int = _envi("SB_PLAYBACK_RATE", 22050)
    playback_block: int = 512               # frames per output callback

    # Half-duplex gate: hard-mute the mic while the head is talking.
    # Set False only if you trust your AEC (ReSpeaker v2.0 hardware AEC).
    mute_mic_while_speaking: bool = _envb("SB_HALF_DUPLEX", True)
    # Extra time to keep the mic gated after audio stops, lets the room decay.
    mic_gate_tail_ms: int = _envi("SB_GATE_TAIL_MS", 250)

    output_gain: float = _envf("SB_OUTPUT_GAIN", 1.0)


# ------------------------------------------------------------------ vad ----

@dataclass
class VADConfig:
    aggressiveness: int = _envi("SB_VAD_AGGRESSIVENESS", 3)   # 0..3, 3 = strictest
    # A party is loud. This RMS floor is the real noise gate; VAD alone is not
    # enough. Calibrate with tools/calibrate_noise.py at the actual party volume.
    rms_floor: float = _envf("SB_RMS_FLOOR", 450.0)
    start_frames: int = _envi("SB_VAD_START_FRAMES", 5)       # ~100ms of speech to open
    end_frames: int = _envi("SB_VAD_END_FRAMES", 35)          # ~700ms of silence to close
    max_utterance_s: float = _envf("SB_MAX_UTTERANCE_S", 12.0)
    min_utterance_s: float = _envf("SB_MIN_UTTERANCE_S", 0.4)
    preroll_ms: int = _envi("SB_PREROLL_MS", 300)             # audio kept from before trigger


# --------------------------------------------------------------- vision ----

@dataclass
class VisionConfig:
    enabled: bool = _envb("SB_VISION", True)
    # Compiled for the Hailo-8L on the AI HAT+ (h8l); ships with hailo-all.
    model: str = _env("SB_HAILO_MODEL", "/usr/share/hailo-models/yolov8s_h8l.hef")
    threshold: float = _envf("SB_DETECT_THRESHOLD", 0.45)
    # A guest counts as "present" once their box is this fraction of frame height.
    # Keeps the head from greeting someone across the yard.
    min_box_height: float = _envf("SB_MIN_BOX_HEIGHT", 0.30)
    # Consecutive frames required before we believe someone arrived / left.
    arrive_frames: int = 6
    depart_frames: int = 25
    # Send one still to the model so it can comment on the costume.
    costume_vision: bool = _envb("SB_COSTUME_VISION", True)
    # Live camera view with the detector's boxes (make view). localhost only,
    # reached through an SSH tunnel. 0 = off.
    liveview_port: int = _envi("SB_LIVEVIEW_PORT", 8081)


# --------------------------------------------------------------- servos ----

@dataclass
class ServoLimits:
    channel: int
    min_deg: float
    max_deg: float
    rest_deg: float
    invert: bool = False


@dataclass
class ServoConfig:
    enabled: bool = _envb("SB_SERVOS", True)
    i2c_address: int = 0x40
    pwm_freq: int = 50
    # Calibrate these with tools/servo_jog.py BEFORE bolting into the skull.
    # Jaw closed must sit a degree or two short of where the jaw physically
    # shuts: past that stop the servo stalls and grinds for as long as it
    # holds. On this skull the stop is ~82 deg (81 grinds), so closed = 83.
    jaw: ServoLimits = field(default_factory=lambda: ServoLimits(
        channel=0,
        min_deg=_envf("SB_JAW_CLOSED_DEG", 83.0),
        max_deg=_envf("SB_JAW_OPEN_DEG", 125.0),
        rest_deg=_envf("SB_JAW_CLOSED_DEG", 83.0)))
    pan: ServoLimits = field(default_factory=lambda: ServoLimits(
        channel=1, min_deg=45.0, max_deg=135.0, rest_deg=90.0))
    tilt: ServoLimits = field(default_factory=lambda: ServoLimits(
        channel=2, min_deg=70.0, max_deg=110.0, rest_deg=90.0))
    # Smoothing for pan/tilt tracking. 0 = no movement, 1 = instant snap.
    track_alpha: float = _envf("SB_TRACK_ALPHA", 0.12)
    # Jaw responsiveness. Higher = snappier, more chatter. 0.55 looks organic.
    jaw_alpha: float = _envf("SB_JAW_ALPHA", 0.55)
    # LED eyes on a spare channel, driven from its PWM pin (through a MOSFET if
    # the pin's 3.3 V can't light them). -1 disables.
    eyes_channel: int = _envi("SB_EYES_CHANNEL", 15)
    # One full fade, dark -> bright -> dark, in seconds.
    eyes_period_s: float = _envf("SB_EYES_PERIOD_S", 2.0)


# ---------------------------------------------------------------- brain ----

@dataclass
class BrainConfig:
    # Picked by measured first-sentence latency vs. humor on the Pi; see README.
    model: str = _env("SB_MODEL", "claude-sonnet-5-5")
    # Fastest no-reasoning settings differ per model: "between_tools" turns
    # thinking off on Sonnet 5.5, Sonnet 5 wants "disabled", and Haiku 4.5
    # takes neither field (set both empty).
    thinking: str = _env("SB_THINKING", "between_tools")
    effort: str = _env("SB_EFFORT", "low")
    max_tokens: int = _envi("SB_MAX_TOKENS", 160)   # short answers = low latency
    # Anthropic's server-side web search: Claude looks things up itself, the Pi
    # needs nothing beyond its normal connection. Needs web search allowed in the
    # org's Console settings. Costs a per-search fee and a few seconds per search.
    web_search: bool = _envb("SB_WEB_SEARCH", True)
    web_search_max_uses: int = _envi("SB_WEB_SEARCH_MAX_USES", 1)   # per reply
    # The basic tool. web_search_20260209 filters results by running code behind
    # the scenes: slower, ~4x the input tokens, and Claude narrates the hidden
    # step out loud ("let me look at the result shape"), which is wrong for a
    # talking head.
    web_search_tool: str = _env("SB_WEB_SEARCH_TOOL", "web_search_20250305")
    # Search queries and results count against the reply's output budget.
    web_max_tokens: int = _envi("SB_WEB_MAX_TOKENS", 400)
    # Comma-separated domains the search must never use.
    web_blocked_domains: str = _env("SB_WEB_BLOCKED_DOMAINS", "")
    history_turns: int = _envi("SB_HISTORY_TURNS", 8)
    # How long a conversation stays "warm" before the head forgets this guest.
    conversation_timeout_s: float = _envf("SB_CONVO_TIMEOUT_S", 45.0)
    # How many past greetings it remembers, to spot repeat costumes. 0 = off.
    greeting_memory: int = _envi("SB_GREETING_MEMORY", 5)
    persona_file: str = _env("SB_PERSONA", "persona.md")


# ------------------------------------------------------------ stt / tts ----

@dataclass
class STTConfig:
    backend: str = _env("SB_STT", "openai")          # openai | faster-whisper
    openai_model: str = _env("SB_STT_MODEL", "gpt-4o-mini-transcribe")
    local_model: str = _env("SB_STT_LOCAL_MODEL", "small.en")
    language: str = "en"


@dataclass
class TTSConfig:
    backend: str = _env("SB_TTS", "elevenlabs")      # elevenlabs | openai | piper
    voice_id: str = _env("SB_VOICE_ID", "onwK4e9ZLuTAKqWW03F9")
    model_id: str = _env("SB_TTS_MODEL", "eleven_flash_v2_5")
    openai_voice: str = _env("SB_OPENAI_VOICE", "onyx")
    # Speaking rate for every backend: 1.0 normal, 0.8 slower and more
    # deliberate, 1.2 brisker. Kokoro/OpenAI accept 0.25-4.0; keep it near 1.
    speed: float = _envf("SB_TTS_SPEED", 1.0)
    # Any OpenAI-compatible server works here (set OPENAI_BASE_URL). Local
    # speaches/Kokoro: SB_OPENAI_TTS_MODEL=speaches-ai/Kokoro-82M-v1.0-ONNX.
    openai_model: str = _env("SB_OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
    # Rate the server returns PCM at. OpenAI is fixed at 24000; speaches honors
    # a sample_rate request, so set this to SB_PLAYBACK_RATE to skip resampling.
    openai_rate: int = _envi("SB_OPENAI_TTS_RATE", 24000)
    piper_model: str = _env("SB_PIPER_MODEL", "/opt/piper/en_US-lessac-medium.onnx")
    stability: float = _envf("SB_TTS_STABILITY", 0.45)
    similarity: float = _envf("SB_TTS_SIMILARITY", 0.75)


# ----------------------------------------------------------------- root ----

@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    servos: ServoConfig = field(default_factory=ServoConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)

    # Seconds of silence from a guest before the head gives up and idles.
    listen_timeout_s: float = _envf("SB_LISTEN_TIMEOUT_S", 45.0)
    # Don't re-greet the same person over and over.
    greet_cooldown_s: float = _envf("SB_GREET_COOLDOWN_S", 30.0)
    # Said mid-conversation, this ends the chat ("As you wish.") and the head
    # goes back to waiting for the wake phrase. Empty = off.
    quiet_phrase: str = _env("SB_QUIET_PHRASE", "quiet bones")
    # Say "hello, my name is Bones" and the time whenever the bot starts up.
    announce_startup: bool = _envb("SB_ANNOUNCE_STARTUP", True)
    # Stay quiet until someone says this (matched on the STT transcript), instead
    # of greeting whoever the camera sees. Empty = greet on sight.
    wake_phrase: str = _env("SB_WAKE_PHRASE", "hello bones")
    # Replies written and spoken ahead of time, so they play with no wait: this
    # many intros for "hello bones", and this many jokes for any request that
    # mentions a joke. The queues refill in the background while he is idle. 0 = off.
    preload_intros: int = _envi("SB_PRELOAD_INTROS", 5)
    preload_jokes: int = _envi("SB_PRELOAD_JOKES", 5)
    log_level: str = _env("SB_LOG_LEVEL", "INFO")


CONFIG = Config()
