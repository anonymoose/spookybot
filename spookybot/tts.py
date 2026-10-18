"""Text to speech, streamed as raw PCM.

Raw PCM (not MP3) is a deliberate choice. MP3 would need decoding before the
jaw could read its envelope, which adds both latency and a decode buffer that
desynchronizes mouth from sound. Requesting PCM at the playback rate means the
bytes off the wire go straight to the DAC and straight to the jaw.
"""
from __future__ import annotations

import logging
import os
import subprocess
import time
from typing import Iterator

import numpy as np

from .config import TTSConfig

log = logging.getLogger(__name__)


class Speech:
    def __init__(self, cfg: TTSConfig, sample_rate: int) -> None:
        self.cfg = cfg
        self.sample_rate = sample_rate
        if cfg.backend == "elevenlabs":
            self._api_key = os.environ["ELEVENLABS_API_KEY"]
            import requests
            self._requests = requests
        elif cfg.backend == "openai":
            from openai import OpenAI
            # Fail fast so the local fallback kicks in while the guest is still there.
            self._client = OpenAI(timeout=10.0, max_retries=0)
        elif cfg.backend == "piper":
            if not os.path.exists(cfg.piper_model):
                raise FileNotFoundError(f"Piper model not found: {cfg.piper_model}")
        else:
            raise ValueError(f"unknown TTS backend: {cfg.backend}")

    def stream(self, text: str) -> Iterator[bytes]:
        log.info("VOICE TTS -> %s: %r", self._describe(), text)
        t0 = time.monotonic()
        first_ms = None
        nbytes = 0
        for chunk in self._stream(text):
            if first_ms is None:
                first_ms = (time.monotonic() - t0) * 1000
            nbytes += len(chunk)
            yield chunk
        log.info("VOICE TTS done: %.1fs of audio, first audio after %s, %.0f ms total",
                 nbytes / 2 / self.sample_rate,
                 f"{first_ms:.0f} ms" if first_ms is not None else "never",
                 (time.monotonic() - t0) * 1000)

    def _describe(self) -> str:
        if self.cfg.backend == "elevenlabs":
            return f"elevenlabs {self.cfg.model_id}"
        if self.cfg.backend == "openai":
            return f"{self.cfg.openai_model} voice={self.cfg.openai_voice}"
        return "piper"

    def _stream(self, text: str) -> Iterator[bytes]:
        try:
            if self.cfg.backend == "elevenlabs":
                yield from self._elevenlabs(text)
            elif self.cfg.backend == "openai":
                yield from self._openai(text)
            else:
                yield from self._piper(text)
        except Exception as exc:
            log.error("VOICE TTS failed (%s); trying local Piper", exc)
            try:
                yield from self._piper(text)
            except Exception as exc2:
                log.error("VOICE local TTS also failed: %s", exc2)

    # -- backends -----------------------------------------------------------

    def _elevenlabs(self, text: str) -> Iterator[bytes]:
        url = (f"https://api.elevenlabs.io/v1/text-to-speech/{self.cfg.voice_id}"
               f"/stream?output_format=pcm_{self.sample_rate}")
        resp = self._requests.post(
            url,
            headers={"xi-api-key": self._api_key, "Content-Type": "application/json"},
            json={
                "text": text,
                "model_id": self.cfg.model_id,
                "voice_settings": {
                    "stability": self.cfg.stability,
                    "similarity_boost": self.cfg.similarity,
                },
            },
            stream=True,
            timeout=30,
        )
        resp.raise_for_status()
        for chunk in resp.iter_content(chunk_size=4096):
            if chunk:
                yield chunk

    def _openai(self, text: str) -> Iterator[bytes]:
        rate = self.cfg.openai_rate
        # OpenAI itself only speaks 24 kHz and rejects unknown fields, so only
        # ask for a rate when pointed at a server that can honor it.
        extra = {"sample_rate": rate} if rate != 24000 else None
        resampler = _Resampler(rate, self.sample_rate) if rate != self.sample_rate else None
        with self._client.audio.speech.with_streaming_response.create(
            model=self.cfg.openai_model,
            voice=self.cfg.openai_voice,
            input=text,
            response_format="pcm",  # signed 16-bit mono
            extra_body=extra,
        ) as resp:
            for chunk in resp.iter_bytes(4096):
                if resampler is not None:
                    chunk = resampler.feed(chunk)
                if chunk:
                    yield chunk

    def _piper(self, text: str) -> Iterator[bytes]:
        proc = subprocess.Popen(
            ["piper", "--model", self.cfg.piper_model, "--output-raw"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        proc.stdin.write(text.encode("utf-8"))
        proc.stdin.close()
        while True:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            yield chunk
        proc.wait()


class _Resampler:
    """Streaming linear resampler for 16-bit mono PCM.

    Playback runs at one fixed rate, so a backend that returns a different rate
    would otherwise play pitched and slowed. Linear interpolation is plenty for
    speech, and carrying one sample across chunks keeps the seams click-free.
    """

    def __init__(self, src_rate: int, dst_rate: int) -> None:
        self.step = src_rate / dst_rate
        self.pos = 0.0
        self.prev = np.zeros(0, dtype=np.float32)
        self.odd = b""

    def feed(self, chunk: bytes) -> bytes:
        data = self.odd + chunk
        n = len(data) // 2 * 2
        self.odd = data[n:]
        x = np.concatenate([self.prev, np.frombuffer(data[:n], dtype=np.int16)
                            .astype(np.float32)])
        if len(x) < 2:
            self.prev = x
            return b""
        last = len(x) - 1
        idx = np.arange(self.pos, last, self.step)
        out = np.interp(idx, np.arange(len(x)), x)
        # Re-base the next read position so x[last] becomes index 0 next time.
        self.pos = (idx[-1] + self.step if len(idx) else self.pos) - last
        self.prev = x[last:]
        return np.round(out).astype(np.int16).tobytes()
