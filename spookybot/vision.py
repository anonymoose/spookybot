"""Camera + Hailo AI HAT: presence detection, head tracking, and costume stills.

The camera (IMX219) has no on-board inference, so the ISP hands a small copy of
each frame straight to the Hailo, which runs YOLOv8 with NMS on-chip. The Pi
receives boxes rather than running a detector itself. That is what makes this
affordable to run continuously alongside audio — the CPU stays free for VAD
and playback instead of burning cores on a detector.

Without the Hailo (HAT removed, cable loose) the camera still runs in
camera-only mode: costume stills work, presence detection doesn't, and the
orchestrator switches to voice-activated arrivals. Neither a missing HAT nor a
missing camera takes the bot down.
"""
from __future__ import annotations

import io
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

log = logging.getLogger(__name__)

try:
    from picamera2 import Picamera2
    from picamera2.devices import Hailo
    _HAVE_CAMERA = True
except Exception:  # pragma: no cover
    _HAVE_CAMERA = False

# The Hailo model zoo detectors are trained on COCO-80, where person is class 0.
PERSON_CLASS = 0

# How often the detection loop reports that it's alive and what it sees, so
# "is the camera even working?" is answered by the log rather than a test run.
STATUS_INTERVAL_S = 30.0


@dataclass
class Guest:
    present: bool = False
    count: int = 0
    center_x: float = 0.5
    center_y: float = 0.5
    box_height: float = 0.0
    last_seen: float = 0.0


class Vision:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.guest = Guest()
        self._lock = threading.Lock()
        self._running = False
        self._picam = None
        self._hailo = None
        # True only while the Hailo is producing detections; False means guest
        # presence is unknown, not that nobody is there.
        self.detecting = False
        self._hits = 0
        self._misses = 0
        self._frames = 0
        self._errors = 0
        self._last_error = ""
        self._best_score = 0.0
        self._status_t0 = time.monotonic()
        # For the live view: every person box from the latest frame (rejected
        # ones carry a reason), a rolling frame clock, and a caller-supplied
        # status line (the orchestrator adds what the bot is doing).
        self._dets: list = []
        self._frame_times: deque = deque(maxlen=30)
        self._render_lock = threading.Lock()
        self._liveview = None
        self.status_fn = None

        if not cfg.enabled:
            log.info("VISION disabled by config")
            return
        if not _HAVE_CAMERA:
            log.warning("VISION picamera2/Hailo unavailable — vision disabled")
            return

        try:
            self._hailo = Hailo(self.cfg.model)
        except Exception as exc:
            log.warning("VISION Hailo unavailable (%s) — camera-only mode: costume "
                        "stills, no presence detection", exc)
            self._hailo = None

        try:
            self._start_camera()
        except Exception as exc:
            log.error("VISION camera failed to start (%s) — vision disabled", exc)
            self.close()
            self._picam = None
            self._hailo = None
            return

        if self._hailo is not None:
            self.detecting = True
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True,
                                            name="vision")
            self._thread.start()
            log.info("VISION detection started: model %s, threshold %.2f, "
                     "min box height %.2f", self.cfg.model, self.cfg.threshold,
                     self.cfg.min_box_height)

        if self._picam is not None and cfg.liveview_port:
            try:
                from . import liveview
                self._liveview = liveview.start(self.render_overlay,
                                                cfg.liveview_port)
            except Exception as exc:
                log.warning("VISION live view unavailable: %s", exc)

    def _start_camera(self) -> None:
        self._picam = Picamera2()
        streams: dict[str, dict] = {"main": {"size": (1280, 720)}}
        if self._hailo is not None:
            model_h, model_w, _ = self._hailo.get_input_shape()
            # The ISP scales the full field of view into the model's square
            # input. The squash doesn't matter: boxes come back normalized, so
            # they map straight onto the main stream. libcamera names formats
            # by register order, so "BGR888" is R,G,B in memory — what YOLO wants.
            streams["lores"] = {"size": (model_w, model_h), "format": "BGR888"}
        config = self._picam.create_preview_configuration(**streams, buffer_count=8)
        self._picam.configure(config)
        self._picam.start()
        time.sleep(1.0)

    def _loop(self) -> None:
        while self._running:
            try:
                frame = self._picam.capture_array("lores")
                self._process(self._hailo.run(frame))
            except Exception as exc:
                self._errors += 1
                self._last_error = str(exc)
                log.debug("VISION frame failed: %s", exc)
                time.sleep(0.1)
            self._report_status()

    def _report_status(self) -> None:
        now = time.monotonic()
        elapsed = now - self._status_t0
        if elapsed < STATUS_INTERVAL_S:
            return
        with self._lock:
            present, count = self.guest.present, self.guest.count
        errors = (f", {self._errors} failed frames (last: {self._last_error})"
                  if self._errors else "")
        log.info("VISION status: %.0f fps, %s, best person score %.2f%s",
                 self._frames / elapsed,
                 f"guest present ({count} in view)" if present else "nobody present",
                 self._best_score, errors)
        self._frames = self._errors = 0
        self._best_score = 0.0
        self._status_t0 = now

    def _process(self, outputs) -> None:
        # Hailo NMS output: one array per class of [y0, x0, y1, x1, score] rows,
        # normalized. Only the person list matters.
        people = []
        dets = []
        for det in outputs[PERSON_CLASS]:
            y0, x0, y1, x1, score = (float(v) for v in det[:5])
            self._best_score = max(self._best_score, score)
            height = abs(y1 - y0)
            reason = ("low" if score < self.cfg.threshold
                      else "small" if height < self.cfg.min_box_height else "")
            dets.append((y0, x0, y1, x1, score, reason))
            if reason:
                continue
            people.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0, height, score))

        self._frames += 1
        self._frame_times.append(time.monotonic())
        with self._lock:
            self._dets = dets
        if people:
            # Track the nearest guest — largest box wins.
            people.sort(key=lambda p: p[2], reverse=True)
            cx, cy, h, score = people[0]
            self._hits += 1
            self._misses = 0
            arrived = False
            with self._lock:
                self.guest.count = len(people)
                self.guest.center_x = cx
                self.guest.center_y = cy
                self.guest.box_height = h
                self.guest.last_seen = time.monotonic()
                if self._hits >= self.cfg.arrive_frames and not self.guest.present:
                    self.guest.present = arrived = True
            if arrived:
                log.info("VISION guest arrived: %d in view, nearest at (%.2f, %.2f), "
                         "box height %.2f, score %.2f", len(people), cx, cy, h, score)
        else:
            self._misses += 1
            self._hits = 0
            if self._misses >= self.cfg.depart_frames:
                left = False
                with self._lock:
                    left = self.guest.present
                    self.guest.present = False
                    self.guest.count = 0
                if left:
                    log.info("VISION guest left: nobody in view for %d frames",
                             self._misses)

    # -- public -------------------------------------------------------------

    def snapshot(self) -> Guest:
        with self._lock:
            return Guest(**vars(self.guest))

    def render_overlay(self, quality: int = 70) -> bytes | None:
        """The current frame with detection boxes and a status line drawn on.
        Called by the live view, once per frame sent to each viewer."""
        if self._picam is None:
            return None
        from .liveview import draw_overlay
        try:
            with self._render_lock:
                image = self._picam.capture_image("main").convert("RGB")
            image.thumbnail((960, 540))
            with self._lock:
                dets = list(self._dets)
                present, count = self.guest.present, self.guest.count
            times = list(self._frame_times)
            fps = ((len(times) - 1) / (times[-1] - times[0])
                   if len(times) > 1 and times[-1] > times[0] else 0.0)
            parts = [f"GUEST x{count}" if present else "nobody",
                     f"{fps:.0f} fps" if self.detecting else "no detector",
                     f"threshold {self.cfg.threshold:.2f}, min height "
                     f"{self.cfg.min_box_height:.2f}"]
            if self.status_fn is not None:
                try:
                    parts.append(self.status_fn())
                except Exception:
                    pass
            draw_overlay(image, dets, "  |  ".join(parts))
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=quality)
            return buf.getvalue()
        except Exception as exc:
            log.debug("VISION live view frame failed: %s", exc)
            return None

    def capture_jpeg(self, quality: int = 70) -> bytes | None:
        """One still for costume commentary. Downscaled — the model does not
        need 720p to tell a witch from a pirate, and small images are faster."""
        if self._picam is None:
            return None
        try:
            image = self._picam.capture_image("main")
            image.thumbnail((640, 640))
            buf = io.BytesIO()
            image.convert("RGB").save(buf, format="JPEG", quality=quality)
            jpeg = buf.getvalue()
            log.info("VISION costume still captured: %d KB", len(jpeg) // 1024)
            return jpeg
        except Exception as exc:
            log.warning("VISION costume still failed: %s", exc)
            return None

    def close(self) -> None:
        if self._liveview is not None:
            self._liveview.shutdown()
            logging.getLogger().removeHandler(self._liveview.log_buffer)
            self._liveview = None
        if self.detecting:
            log.info("VISION detection stopped")
        self._running = False
        self.detecting = False
        if self._picam is not None:
            for step in (self._picam.stop, self._picam.close):
                try:
                    step()
                except Exception:
                    pass
        if self._hailo is not None:
            try:
                self._hailo.close()
            except Exception:
                pass
