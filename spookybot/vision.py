"""Camera + Hailo AI HAT: presence detection, head tracking, and costume stills.

The camera (IMX219) has no on-board inference, so the ISP hands a small copy of
each frame straight to the Hailo, which runs YOLOv8 with NMS on-chip. The Pi
receives boxes rather than running a detector itself. That is what makes this
affordable to run continuously alongside audio — the CPU stays free for VAD
and playback instead of burning cores on a detector.
"""
from __future__ import annotations

import io
import logging
import threading
import time
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
        self._hits = 0
        self._misses = 0

        if not cfg.enabled:
            log.info("Vision disabled by config")
            return
        if not _HAVE_CAMERA:
            log.warning("picamera2/Hailo unavailable — vision disabled")
            return

        self._start()

    def _start(self) -> None:
        self._hailo = Hailo(self.cfg.model)
        model_h, model_w, _ = self._hailo.get_input_shape()

        self._picam = Picamera2()
        config = self._picam.create_preview_configuration(
            main={"size": (1280, 720)},
            # The ISP scales the full field of view into the model's square
            # input. The squash doesn't matter: boxes come back normalized, so
            # they map straight onto the main stream. libcamera names formats
            # by register order, so "BGR888" is R,G,B in memory — what YOLO wants.
            lores={"size": (model_w, model_h), "format": "BGR888"},
            buffer_count=8,
        )
        self._picam.configure(config)
        self._picam.start()
        time.sleep(1.0)

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="vision")
        self._thread.start()
        log.info("Vision running with model %s", self.cfg.model)

    def _loop(self) -> None:
        while self._running:
            try:
                frame = self._picam.capture_array("lores")
                self._process(self._hailo.run(frame))
            except Exception as exc:
                log.debug("vision frame failed: %s", exc)
                time.sleep(0.1)

    def _process(self, outputs) -> None:
        # Hailo NMS output: one array per class of [y0, x0, y1, x1, score] rows,
        # normalized. Only the person list matters.
        people = []
        for det in outputs[PERSON_CLASS]:
            y0, x0, y1, x1, score = (float(v) for v in det[:5])
            if score < self.cfg.threshold:
                continue
            height = abs(y1 - y0)
            if height < self.cfg.min_box_height:
                continue
            people.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0, height))

        if people:
            # Track the nearest guest — largest box wins.
            people.sort(key=lambda p: p[2], reverse=True)
            cx, cy, h = people[0]
            self._hits += 1
            self._misses = 0
            with self._lock:
                self.guest.count = len(people)
                self.guest.center_x = cx
                self.guest.center_y = cy
                self.guest.box_height = h
                self.guest.last_seen = time.monotonic()
                if self._hits >= self.cfg.arrive_frames:
                    self.guest.present = True
        else:
            self._misses += 1
            self._hits = 0
            if self._misses >= self.cfg.depart_frames:
                with self._lock:
                    self.guest.present = False
                    self.guest.count = 0

    # -- public -------------------------------------------------------------

    def snapshot(self) -> Guest:
        with self._lock:
            return Guest(**vars(self.guest))

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
            return buf.getvalue()
        except Exception as exc:
            log.debug("snapshot failed: %s", exc)
            return None

    def close(self) -> None:
        self._running = False
        if self._picam is not None:
            try:
                self._picam.stop()
            except Exception:
                pass
        if self._hailo is not None:
            try:
                self._hailo.close()
            except Exception:
                pass
