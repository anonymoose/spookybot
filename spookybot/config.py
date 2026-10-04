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
    model: str = _env(
        "SB_IMX500_MODEL",
        "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk",
    )
    labels: str = _env("SB_IMX500_LABELS", "/usr/share/imx500-models/coco_labels.txt")
    threshold: float = _envf("SB_DETECT_THRESHOLD", 0.45)
    person_label: str = "person"
    # A guest counts as "present" once their box is this fraction of frame height.
    # Keeps the head from greeting someone across the yard.
    min_box_height: float = _envf("SB_MIN_BOX_HEIGHT", 0.30)
    # Consecutive frames required before we believe someone arrived / left.
    arrive_frames: int = 6
    depart_frames: int = 25
    # Send one still to the model so it can comment on the costume.
    costume_vision: bool = _envb("SB_COSTUME_VISION", True)


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
    jaw: ServoLimits = field(default_factory=lambda: ServoLimits(
        channel=0, min_deg=78.0, max_deg=125.0, rest_deg=78.0))
    pan: ServoLimits = field(default_factory=lambda: ServoLimits(
        channel=1, min_deg=45.0, max_deg=135.0, rest_deg=90.0))
    tilt: ServoLimits = field(default_factory=lambda: ServoLimits(
        channel=2, min_deg=70.0, max_deg=110.0, rest_deg=90.0))
    # Smoothing for pan/tilt tracking. 0 = no movement, 1 = instant snap.
    track_alpha: float = _envf("SB_TRACK_ALPHA", 0.12)
    # Jaw responsiveness. Higher = snappier, more chatter. 0.55 looks organic.
    jaw_alpha: float = _envf("SB_JAW_ALPHA", 0.55)


# ---------------------------------------------------------------- brain ----

@dataclass
class BrainConfig:
    model: str = _env("SB_MODEL", "claude-sonnet-5")
    max_tokens: int = _envi("SB_MAX_TOKENS", 160)   # short answers = low latency
    temperature: float = _envf("SB_TEMPERATURE", 1.0)
    history_turns: int = _envi("SB_HISTORY_TURNS", 8)
    # How long a conversation stays "warm" before the head forgets this guest.
    conversation_timeout_s: float = _envf("SB_CONVO_TIMEOUT_S", 45.0)
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
    listen_timeout_s: float = _envf("SB_LISTEN_TIMEOUT_S", 10.0)
    # Don't re-greet the same person over and over.
    greet_cooldown_s: float = _envf("SB_GREET_COOLDOWN_S", 30.0)
    log_level: str = _env("SB_LOG_LEVEL", "INFO")


CONFIG = Config()
