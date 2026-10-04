"""Microphone capture and speaker playback, with jaw sync.

The important trick lives in `play_pcm`: the same audio callback that hands a
block of samples to the speaker also computes that block's RMS and pushes it to
the jaw. Sync is therefore exact by construction — there is no clock to drift,
because the jaw is driven by the very buffer the DAC is about to play.
"""
from __future__ import annotations

import collections
import logging
import queue
import threading
import time

import numpy as np
import sounddevice as sd

from .config import AudioConfig

log = logging.getLogger(__name__)


def find_device(name_fragment: str, want_input: bool) -> int | None:
    """Match a device by substring so you don't hardcode fragile index numbers."""
    if not name_fragment:
        return None
    key = "max_input_channels" if want_input else "max_output_channels"
    frag = name_fragment.lower()
    for idx, dev in enumerate(sd.query_devices()):
        if dev[key] > 0 and frag in dev["name"].lower():
            log.info("Matched %s device %d: %s", "input" if want_input else "output",
                     idx, dev["name"])
            return idx
    log.warning("No %s device matching %r; falling back to system default",
                "input" if want_input else "output", name_fragment)
    return None


class MicStream:
    """Always-on capture into a ring buffer, with a hard gate for half-duplex.

    Keeping the stream open the whole time (rather than opening it per
    utterance) matters: ALSA device open on the Pi costs 200-400ms, which is
    exactly the window in which a guest says the first word of their sentence.
    """

    def __init__(self, cfg: AudioConfig) -> None:
        self.cfg = cfg
        self.block = int(cfg.sample_rate * cfg.block_ms / 1000)
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self._gated_until = 0.0
        self._lock = threading.Lock()

        self._stream = sd.InputStream(
            samplerate=cfg.sample_rate,
            channels=cfg.channels,
            dtype="int16",
            blocksize=self.block,
            device=find_device(cfg.input_device, want_input=True),
            callback=self._callback,
        )
        self._stream.start()
        log.info("Mic open: %d Hz, %d ms blocks", cfg.sample_rate, cfg.block_ms)

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            log.debug("input status: %s", status)
        with self._lock:
            gated = time.monotonic() < self._gated_until
        if gated:
            return
        try:
            self._q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass  # we're behind; dropping the oldest audio is the right call

    def gate(self, seconds: float) -> None:
        """Mute capture for `seconds`. Used while the head is speaking."""
        with self._lock:
            self._gated_until = max(self._gated_until, time.monotonic() + seconds)

    def ungate(self) -> None:
        with self._lock:
            self._gated_until = 0.0

    def flush(self) -> None:
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return

    def read(self, timeout: float = 0.5) -> np.ndarray | None:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


class Speaker:
    """Streaming PCM playback that drives the jaw from the playing buffer."""

    def __init__(self, cfg: AudioConfig, jaw_callback=None) -> None:
        self.cfg = cfg
        self.jaw_callback = jaw_callback
        self._device = find_device(cfg.output_device, want_input=False)
        self._buffer = collections.deque()
        self._lock = threading.Lock()
        self._finished = threading.Event()
        self._finished.set()
        self._stop = False
        # Rolling peak, used to normalize the envelope into jaw range. TTS
        # loudness varies per sentence; a fixed threshold makes the jaw either
        # barely twitch or hang wide open.
        self._peak = 3000.0

    # -- feeding ------------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            self._buffer.clear()
            self._stop = False
        self._finished.clear()
        self._stream = sd.OutputStream(
            samplerate=self.cfg.playback_rate,
            channels=1,
            dtype="int16",
            blocksize=self.cfg.playback_block,
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()

    def feed(self, pcm: bytes) -> None:
        samples = np.frombuffer(pcm, dtype=np.int16)
        if self.cfg.output_gain != 1.0:
            samples = np.clip(samples.astype(np.float32) * self.cfg.output_gain,
                              -32768, 32767).astype(np.int16)
        with self._lock:
            self._buffer.extend(samples)

    def done_feeding(self) -> None:
        """Signal that no more PCM is coming; playback ends when buffer drains."""
        with self._lock:
            self._stop = True

    def wait(self, timeout: float | None = None) -> bool:
        return self._finished.wait(timeout)

    def stop(self) -> None:
        """Hard stop — used for barge-in."""
        with self._lock:
            self._buffer.clear()
            self._stop = True

    def _callback(self, outdata, frames, time_info, status) -> None:
        if status:
            log.debug("output status: %s", status)
        with self._lock:
            available = len(self._buffer)
            take = min(frames, available)
            chunk = np.array([self._buffer.popleft() for _ in range(take)],
                             dtype=np.int16) if take else np.zeros(0, dtype=np.int16)
            ending = self._stop and available <= frames

        if take < frames:
            chunk = np.concatenate([chunk, np.zeros(frames - take, dtype=np.int16)])
        outdata[:, 0] = chunk

        # Drive the jaw from exactly the samples now going to the DAC.
        if self.jaw_callback is not None:
            rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)) + 1e-6)
            self._peak = max(rms, self._peak * 0.9995)
            self.jaw_callback(self._shape(rms / self._peak))

        if ending:
            self._finished.set()
            raise sd.CallbackStop()

    @staticmethod
    def _shape(x: float) -> float:
        """Map normalized loudness to jaw openness.

        Raw RMS maps badly: speech spends most of its energy in a narrow band,
        so a linear map leaves the jaw fluttering near-closed. The sqrt lifts
        quiet syllables, and the floor keeps the mouth from fully sealing
        between words, which reads as a stutter rather than speech.
        """
        x = max(0.0, min(1.0, x))
        shaped = x ** 0.5
        return 0.0 if shaped < 0.08 else min(1.0, 0.15 + shaped * 0.95)

    def close(self) -> None:
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass
