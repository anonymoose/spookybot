"""Utterance detection: decide when a guest started and stopped talking.

WebRTC's VAD alone is not sufficient at a party. It answers "is this speech?",
and at a Halloween party the answer is almost always yes — it's just other
people's speech. The RMS floor is what actually separates the person leaning
into the prop from the crowd four feet behind them, because loudness falls off
with distance and VAD confidence does not.

Calibrate `rms_floor` at the real party volume, not in a quiet room.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from .config import VADConfig

log = logging.getLogger(__name__)

try:
    import webrtcvad
    _HAVE_VAD = True
except Exception:  # pragma: no cover
    _HAVE_VAD = False


class UtteranceDetector:
    def __init__(self, cfg: VADConfig, sample_rate: int, block_ms: int) -> None:
        self.cfg = cfg
        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self._vad = webrtcvad.Vad(cfg.aggressiveness) if _HAVE_VAD else None
        if not _HAVE_VAD:
            log.warning("webrtcvad unavailable — using RMS-only gating")
        self.preroll_blocks = max(1, int(cfg.preroll_ms / block_ms))

    def _is_speech(self, block: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(block.astype(np.float32) ** 2)))
        if rms < self.cfg.rms_floor:
            return False
        if self._vad is None:
            return True
        try:
            return self._vad.is_speech(block.tobytes(), self.sample_rate)
        except Exception:
            return True

    def capture(self, mic, timeout_s: float, on_speech_start=None) -> np.ndarray | None:
        """Block until a complete utterance is captured, or `timeout_s` elapses.

        Returns int16 mono audio, or None if nobody said anything usable.
        """
        from collections import deque

        preroll: deque = deque(maxlen=self.preroll_blocks)
        collected: list[np.ndarray] = []
        speech_run = 0
        silence_run = 0
        started = False
        deadline = time.monotonic() + timeout_s
        max_blocks = int(self.cfg.max_utterance_s * 1000 / self.block_ms)

        while True:
            if not started and time.monotonic() > deadline:
                return None

            block = mic.read(timeout=0.4)
            if block is None:
                continue

            speech = self._is_speech(block)

            if not started:
                preroll.append(block)
                speech_run = speech_run + 1 if speech else 0
                if speech_run >= self.cfg.start_frames:
                    started = True
                    collected.extend(preroll)
                    if on_speech_start:
                        on_speech_start()
                    log.debug("utterance started")
                continue

            collected.append(block)
            silence_run = 0 if speech else silence_run + 1

            if silence_run >= self.cfg.end_frames or len(collected) >= max_blocks:
                audio = np.concatenate(collected)
                duration = len(audio) / self.sample_rate
                if duration < self.cfg.min_utterance_s:
                    log.debug("utterance too short (%.2fs), discarding", duration)
                    return None
                log.debug("utterance captured: %.2fs", duration)
                return audio


def measure_noise_floor(mic, seconds: float = 5.0) -> dict:
    """Helper for tools/calibrate_noise.py."""
    values: list[float] = []
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        block = mic.read(timeout=0.5)
        if block is None:
            continue
        values.append(float(np.sqrt(np.mean(block.astype(np.float32) ** 2))))
    arr = np.array(values)
    return {
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
        "suggested_floor": float(np.percentile(arr, 95) * 1.6),
    }
