"""Speech to text. Hosted by default, local fallback for when party wifi dies.

Party wifi dying is not a hypothetical. Have the local backend installed and
tested before Halloween, even if you plan to run hosted.
"""
from __future__ import annotations

import io
import logging
import wave

import numpy as np

from .config import STTConfig

log = logging.getLogger(__name__)


def pcm_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


class Transcriber:
    def __init__(self, cfg: STTConfig, sample_rate: int) -> None:
        self.cfg = cfg
        self.sample_rate = sample_rate
        self._local = None
        if cfg.backend == "openai":
            from openai import OpenAI
            self._client = OpenAI()
        elif cfg.backend == "faster-whisper":
            self._load_local()
        else:
            raise ValueError(f"unknown STT backend: {cfg.backend}")

    def _load_local(self) -> None:
        from faster_whisper import WhisperModel
        log.info("Loading local Whisper model %s", self.cfg.local_model)
        # int8 on the Pi 5's CPU: small.en runs roughly real-time, which is
        # acceptable for a fallback but noticeably laggier than hosted.
        self._local = WhisperModel(self.cfg.local_model, device="cpu",
                                   compute_type="int8")

    def transcribe(self, audio: np.ndarray) -> str:
        if self.cfg.backend == "openai":
            try:
                return self._transcribe_hosted(audio)
            except Exception as exc:
                log.warning("Hosted STT failed (%s); falling back to local", exc)
                if self._local is None:
                    self._load_local()
                return self._transcribe_local(audio)
        return self._transcribe_local(audio)

    def _transcribe_hosted(self, audio: np.ndarray) -> str:
        wav = pcm_to_wav_bytes(audio, self.sample_rate)
        fh = io.BytesIO(wav)
        fh.name = "utterance.wav"
        result = self._client.audio.transcriptions.create(
            model=self.cfg.openai_model,
            file=fh,
            language=self.cfg.language,
            # Priming the decoder with in-domain vocabulary measurably cuts
            # errors on costume nouns, which is most of what guests will say.
            prompt="Halloween party, costumes, ghosts, witches, zombies, candy.",
        )
        return (result.text or "").strip()

    def _transcribe_local(self, audio: np.ndarray) -> str:
        samples = audio.astype(np.float32) / 32768.0
        segments, _ = self._local.transcribe(
            samples, language=self.cfg.language, beam_size=1, vad_filter=False
        )
        return " ".join(seg.text for seg in segments).strip()
