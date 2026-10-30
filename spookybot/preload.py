"""Replies written and spoken ahead of time, for the requests that come up most.

Waiting for Claude to write a reply and the speech server to voice it costs a
couple of seconds. For "hello bones" (the intro) and for jokes the answer can
be prepared in advance: Claude writes a few, each sentence is synthesized and
held in the speech cache, and when the request arrives the reply plays at once.
The queues refill in the background, only while the bot is idle so they never
compete with a live conversation.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Callable

from .brain import split_sentences

log = logging.getLogger(__name__)

# (user_text, note) for a kind of reply, given the replies to steer away from.
Prompt = Callable[[list], "tuple[str, str]"]


class Preloader:
    def __init__(self, brain, tts, targets: dict, prompts: dict,
                 idle: Callable[[], bool] = lambda: True,
                 backoff_s: float = 15.0) -> None:
        self.brain, self.tts = brain, tts
        self.targets = {k: n for k, n in targets.items() if n > 0}
        self.prompts = prompts
        self.idle = idle
        self.backoff_s = backoff_s
        self._queues = {k: deque() for k in self.targets}
        self._recent = {k: deque(maxlen=10) for k in self.targets}   # already used
        self._lock = threading.Lock()
        self._need = threading.Event()
        self._running = False

    # -- using the queues ---------------------------------------------------

    def take(self, kind: str):
        """The next prepared reply as a list of sentences, or None if empty."""
        with self._lock:
            queue = self._queues.get(kind)
            if not queue:
                return None
            sentences = queue.popleft()
            self._recent[kind].append(" ".join(sentences))
            left = len(queue)
        log.info("PRELOAD %s taken, %d left", kind, left)
        self._need.set()
        return sentences

    def clear(self) -> None:
        """Throw away every prepared reply (they were written with an old
        persona.md) and wake the refiller to write new ones."""
        with self._lock:
            stale = [sentences for q in self._queues.values() for sentences in q]
            for q in self._queues.values():
                q.clear()
        for sentences in stale:
            for sentence in sentences:
                self.tts.discard(sentence)
        log.info("PRELOAD cleared %d prepared replies; rewriting them", len(stale))
        self._need.set()

    def counts(self) -> dict:
        with self._lock:
            return {k: len(q) for k, q in self._queues.items()}

    # -- filling them -------------------------------------------------------

    def _avoid(self, kind: str) -> list:
        with self._lock:
            return ([" ".join(s) for s in self._queues[kind]]
                    + list(self._recent[kind]))

    def _make(self, kind: str):
        user_text, note = self.prompts[kind](self._avoid(kind))
        text = self.brain.generate(user_text, note)
        if not text:
            return None
        sentences = split_sentences(text)
        for i, sentence in enumerate(sentences):
            if not self.tts.warm(sentence):
                for done in sentences[:i]:
                    self.tts.discard(done)
                return None
        return sentences

    def fill_once(self, kinds=None, force: bool = False) -> int:
        """Top the queues up to their targets (all of them, or just `kinds`).
        Stops early on a failure or when the bot stops being idle. Returns how
        many replies were prepared."""
        made = 0
        for kind, target in self.targets.items():
            if kinds is not None and kind not in kinds:
                continue
            while True:
                with self._lock:
                    if len(self._queues[kind]) >= target:
                        break
                if not force and not self.idle():
                    return made
                sentences = self._make(kind)
                if sentences is None:
                    return made
                with self._lock:
                    self._queues[kind].append(sentences)
                    have = len(self._queues[kind])
                made += 1
                log.info("PRELOAD %s ready %d/%d: %s", kind, have, target,
                         " ".join(sentences)[:70])
        return made

    def fill_initial(self, max_passes: int = 3) -> bool:
        """The first stocking, run to completion on the calling thread (the
        queues are filled in parallel, one thread each, so it takes about as
        long as the slower queue). Retries a failed pass up to `max_passes`
        times. Returns True if every queue is full."""
        for _ in range(max_passes):
            workers = [threading.Thread(target=self.fill_once, args=([kind], True))
                       for kind in self.targets]
            for w in workers:
                w.start()
            for w in workers:
                w.join()
            if self._full():
                return True
            time.sleep(self.backoff_s)
        return self._full()

    def _full(self) -> bool:
        with self._lock:
            return all(len(self._queues[k]) >= n for k, n in self.targets.items())

    def start(self) -> None:
        if not self.targets or self._running:
            return
        self._running = True
        threading.Thread(target=self._run, daemon=True, name="preload").start()
        log.info("PRELOAD stocking %s", ", ".join(
            f"{n} {k}" for k, n in self.targets.items()))

    def stop(self) -> None:
        self._running = False
        self._need.set()

    def _run(self) -> None:
        while self._running:
            if not self.idle():
                self._need.wait(1.0)
                self._need.clear()
                continue
            made = self.fill_once()
            if self._full():
                self._need.wait(60.0)
                self._need.clear()
            elif made == 0:
                # A failure (Claude or the speech server unreachable): back off.
                self._need.wait(self.backoff_s)
                self._need.clear()
