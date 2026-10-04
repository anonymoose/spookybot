"""IMX500 AI Camera: presence detection, head tracking, and costume stills.

Inference runs on the sensor, so the Pi receives boxes rather than frames. That
is what makes this affordable to run continuously alongside audio — the CPU
stays free for VAD and playback instead of burning cores on a detector.
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
    from picamera2.devices import IMX500
    from picamera2.devices.imx500 import NetworkIntrinsics
    _HAVE_CAMERA = True
except Exception:  # pragma: no cover
    _HAVE_CAMERA = False


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
        self._imx500 = None
        self._labels: list[str] = []
        self._hits = 0
        self._misses = 0

        if not cfg.enabled:
            log.info("Vision disabled by config")
            return
        if not _HAVE_CAMERA:
            log.warning("picamera2/IMX500 unavailable — vision disabled")
            return

        self._start()

    def _start(self) -> None:
        self._imx500 = IMX500(self.cfg.model)
        intrinsics = self._imx500.network_intrinsics or NetworkIntrinsics()
        intrinsics.task = "object detection"
        try:
            with open(self.cfg.labels, "r", encoding="utf-8") as fh:
                self._labels = [ln.strip() for ln in fh if ln.strip()]
        except OSError:
            self._labels = intrinsics.labels or []
        intrinsics.update_with_defaults()

        self._picam = Picamera2(self._imx500.camera_num)
        config = self._picam.create_preview_configuration(
            main={"size": (1280, 720)}, buffer_count=8
        )
        self._picam.start(config)
        self._imx500.show_network_fw_progress_bar()
        time.sleep(1.0)

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="vision")
        self._thread.start()
        log.info("Vision running with model %s", self.cfg.model)

    def _loop(self) -> None:
        while self._running:
            try:
                metadata = self._picam.capture_metadata()
                outputs = self._imx500.get_outputs(metadata, add_batch=True)
                if outputs is None:
                    time.sleep(0.03)
                    continue
                self._process(outputs, metadata)
            except Exception as exc:
                log.debug("vision frame failed: %s", exc)
                time.sleep(0.1)

    def _process(self, outputs, metadata) -> None:
        boxes, scores, classes = outputs[0][0], outputs[1][0], outputs[2][0]
        input_w, input_h = self._imx500.get_input_size()

        people = []
        for box, score, cls in zip(boxes, scores, classes):
            if score < self.cfg.threshold:
                continue
            idx = int(cls)
            label = self._labels[idx] if idx < len(self._labels) else str(idx)
            if label != self.cfg.person_label:
                continue
            y0, x0, y1, x1 = box  # IMX500 postprocess emits y-first, normalized
            height = abs(float(y1) - float(y0))
            if height < self.cfg.min_box_height:
                continue
            people.append((
                (float(x0) + float(x1)) / 2.0,
                (float(y0) + float(y1)) / 2.0,
                height,
            ))

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
