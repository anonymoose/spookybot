"""Servo control via PCA9685.

Why a PCA9685 and not Pi GPIO PWM: the Pi 5's software PWM jitters under load,
and a jittering jaw servo buzzes audibly — which, two feet from a microphone,
is a problem. The PCA9685 generates PWM in hardware and costs $6.

POWER: do NOT feed servos from the Pi's 5V rail. A stalled MG996R pulls >2A and
will brown out the Pi mid-sentence. Use a separate 6V 5A supply into the
PCA9685 V+ terminal, and tie its ground to the Pi's ground.
"""
from __future__ import annotations

import logging
import threading
import time

from .config import ServoConfig, ServoLimits

log = logging.getLogger(__name__)

try:
    from adafruit_pca9685 import PCA9685
    from adafruit_motor import servo as _servo
    import board
    import busio
    _HAVE_HW = True
except Exception:  # pragma: no cover - dev machines have no I2C
    _HAVE_HW = False


class _NullServo:
    """Stand-in so the whole rig runs on a laptop with no hardware attached."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.angle = 90.0


class ServoBus:
    """Thread-safe wrapper around the three servos: jaw, pan, tilt."""

    def __init__(self, cfg: ServoConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self._targets: dict[str, float] = {}
        self._current: dict[str, float] = {}
        self._servos: dict[str, object] = {}

        if cfg.enabled and _HAVE_HW:
            i2c = busio.I2C(board.SCL, board.SDA)
            self._pca = PCA9685(i2c, address=cfg.i2c_address)
            self._pca.frequency = cfg.pwm_freq
            make = lambda ch: _servo.Servo(  # noqa: E731
                self._pca.channels[ch], min_pulse=500, max_pulse=2500
            )
        else:
            if cfg.enabled:
                log.warning("Servo libs unavailable — running in simulation mode")
            self._pca = None
            make = lambda ch: _NullServo(f"ch{ch}")  # noqa: E731

        for name, limits in (("jaw", cfg.jaw), ("pan", cfg.pan), ("tilt", cfg.tilt)):
            self._servos[name] = make(limits.channel)
            self._current[name] = limits.rest_deg
            self._targets[name] = limits.rest_deg

        self._limits = {"jaw": cfg.jaw, "pan": cfg.pan, "tilt": cfg.tilt}
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="servos")
        self._thread.start()

    # -- public API ---------------------------------------------------------

    def set_normalized(self, name: str, value: float) -> None:
        """value in 0.0..1.0, mapped into that servo's calibrated range."""
        lim = self._limits[name]
        value = max(0.0, min(1.0, value))
        if lim.invert:
            value = 1.0 - value
        deg = lim.min_deg + (lim.max_deg - lim.min_deg) * value
        with self._lock:
            self._targets[name] = deg

    def set_jaw(self, openness: float) -> None:
        self.set_normalized("jaw", openness)

    def look_at(self, x: float, y: float) -> None:
        """x, y are normalized frame coords (0..1). Center of frame is 0.5, 0.5."""
        self.set_normalized("pan", 1.0 - x)   # camera mirrors the head's view
        self.set_normalized("tilt", 1.0 - y)

    def rest(self) -> None:
        with self._lock:
            for name, lim in self._limits.items():
                self._targets[name] = lim.rest_deg

    def close(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self.rest()
        time.sleep(0.3)
        if self._pca is not None:
            try:
                self._pca.deinit()
            except Exception:
                pass

    # -- internals ----------------------------------------------------------

    def _loop(self) -> None:
        """Runs at 50 Hz — matches the PWM frame rate, no point going faster."""
        period = 1.0 / 50.0
        while self._running:
            t0 = time.perf_counter()
            with self._lock:
                targets = dict(self._targets)
            for name, target in targets.items():
                alpha = self.cfg.jaw_alpha if name == "jaw" else self.cfg.track_alpha
                cur = self._current[name]
                new = cur + (target - cur) * alpha
                if abs(new - cur) > 0.15:
                    self._current[name] = new
                    try:
                        self._servos[name].angle = new
                    except Exception as exc:  # a single bad write shouldn't kill it
                        log.debug("servo write failed on %s: %s", name, exc)
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
