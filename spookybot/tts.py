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
from typing import Iterator

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
            self._client = OpenAI()
        elif cfg.backend == "piper":
            if not os.path.exists(cfg.piper_model):
                raise FileNotFoundError(f"Piper model not found: {cfg.piper_model}")
        else:
            raise ValueError(f"unknown TTS backend: {cfg.backend}")

    def stream(self, text: str) -> Iterator[bytes]:
        try:
            if self.cfg.backend == "elevenlabs":
                yield from self._elevenlabs(text)
            elif self.cfg.backend == "openai":
                yield from self._openai(text)
            else:
                yield from self._piper(text)
        except Exception as exc:
            log.error("TTS failed (%s); trying local Piper", exc)
            try:
                yield from self._piper(text)
            except Exception as exc2:
                log.error("Local TTS also failed: %s", exc2)

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
        with self._client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice=self.cfg.openai_voice,
            input=text,
            response_format="pcm",  # 24 kHz signed 16-bit mono
        ) as resp:
            for chunk in resp.iter_bytes(4096):
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
