"""The conversation layer: Claude, streamed and chunked into sentences.

Latency is the whole game. A guest who waits three seconds walks away. The fix
is to never wait for the full response: stream tokens, cut at sentence
boundaries, and hand each finished sentence to TTS while Claude is still
writing the next one. First audio then lands at roughly first-sentence latency
rather than full-response latency, which in practice is the difference between
~800ms and ~3s.
"""
from __future__ import annotations

import base64
import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field

from .config import BrainConfig

log = logging.getLogger(__name__)

DEFAULT_PERSONA = """\
You are an animatronic talking head at a Halloween party — a theatrical, \
slightly menacing but fundamentally good-humored spirit who has been bolted to \
a table and is making the best of it.

Rules that matter:
- Keep replies to ONE or TWO sentences. You are speaking aloud at a loud party; \
long answers get walked away from.
- Be witty and quick rather than spooky-generic. Never say "mortal" unironically.
- You can see the guest through a camera. If their costume is mentioned in the \
context, react to it specifically — that lands better than any scripted line.
- Your backstory (the wall, the bolts, 1874) is for your introduction only. \
After that, never mention it again.
- If you didn't catch what someone said, say so in character and ask once.
- Never break character to discuss being an AI, a model, or a prop.
- Keep it PG-13. Children will talk to you. No profanity, nothing genuinely \
frightening to a small kid, nothing cruel about anyone's appearance or body.
- Never give out real personal information or claim to know a guest's name \
unless they just told you.
"""

# USD per million tokens (input, output), Anthropic list prices as of 2026-09.
# Cache writes bill at 1.25x input, cache reads at 0.1x. Only used for the
# CHAT log line; a model missing here logs tokens without a cost.
_PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def call_cost(model: str, usage) -> float | None:
    """Dollar cost of one call from its usage block, or None if unpriced."""
    # Longest matching prefix, so "claude-sonnet-5-5" doesn't price as
    # "claude-sonnet-5" and dated IDs still match their family.
    keys = [k for k in _PRICES if model.startswith(k)]
    if not keys:
        return None
    price_in, price_out = _PRICES[max(keys, key=len)]
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    return (usage.input_tokens * price_in
            + cache_write * price_in * 1.25
            + cache_read * price_in * 0.1
            + usage.output_tokens * price_out) / 1_000_000


_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|(?<=[.!?…])$")


@dataclass
class Conversation:
    messages: list = field(default_factory=list)
    last_activity: float = 0.0

    def is_stale(self, timeout_s: float) -> bool:
        return time.monotonic() - self.last_activity > timeout_s

    def touch(self) -> None:
        self.last_activity = time.monotonic()


class Brain:
    def __init__(self, cfg: BrainConfig) -> None:
        from anthropic import Anthropic

        self.cfg = cfg
        self.client = Anthropic()
        self.persona = self._load_persona()
        self.conversation = Conversation()
        # What the head said to its last few arrivals, newest last. Survives
        # reset() so it can notice a costume it has already seen tonight.
        self.recent_greetings: deque[str] = deque(maxlen=cfg.greeting_memory)
        self.calls = 0
        self.spent = 0.0      # dollars since the bot started

    def _load_persona(self) -> str:
        path = self.cfg.persona_file
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read().strip()
            if text:
                log.info("Loaded persona from %s", path)
                return text
        return DEFAULT_PERSONA

    def reset_if_stale(self) -> bool:
        if self.conversation.messages and \
                self.conversation.is_stale(self.cfg.conversation_timeout_s):
            log.info("Conversation timed out; forgetting this guest")
            self.conversation = Conversation()
            return True
        return False

    def reset(self) -> None:
        self.conversation = Conversation()

    def respond(self, user_text: str, image_jpeg: bytes | None = None,
                scene_note: str | None = None):
        """Yield complete sentences as Claude produces them."""
        content: list = []
        if image_jpeg is not None:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.b64encode(image_jpeg).decode("ascii"),
                },
            })
        prefix = f"[{scene_note}]\n" if scene_note else ""
        content.append({"type": "text", "text": f"{prefix}{user_text}"})

        self.conversation.messages.append({"role": "user", "content": content})
        self._trim()

        # A talking head can't afford a silent reasoning pause, and thinking
        # would eat max_tokens. (Sampling params are gone on current models.)
        extra: dict = {}
        if self.cfg.thinking:
            extra["thinking"] = {"type": self.cfg.thinking}
        if self.cfg.effort:
            extra["output_config"] = {"effort": self.cfg.effort}

        log.info("CHAT -> %s: %r%s (%d messages in context)",
                 self.cfg.model, f"{prefix}{user_text}",
                 f" + image {len(image_jpeg) // 1024} KB" if image_jpeg else "",
                 len(self.conversation.messages))

        buffer = ""
        full = ""
        t0 = time.monotonic()
        first_ms = None
        final = None
        try:
            with self.client.messages.stream(
                model=self.cfg.model,
                max_tokens=self.cfg.max_tokens,
                **extra,
                system=self.persona,
                messages=self.conversation.messages,
            ) as stream:
                for token in stream.text_stream:
                    if first_ms is None:
                        first_ms = (time.monotonic() - t0) * 1000
                    buffer += token
                    full += token
                    while True:
                        match = _SENTENCE_END.search(buffer)
                        if not match:
                            break
                        sentence = buffer[:match.end()].strip()
                        buffer = buffer[match.end():]
                        if sentence:
                            yield sentence
                final = stream.get_final_message()
        except Exception as exc:
            log.error("CHAT call failed: %s", exc)
            if not full:
                yield "My wires are crossed. Ask me again."
                return

        if buffer.strip():
            yield buffer.strip()

        self._log_reply(full, final, first_ms, (time.monotonic() - t0) * 1000)

        if full.strip():
            self.conversation.messages.append(
                {"role": "assistant", "content": full.strip()})
        self._trim()
        self.conversation.touch()

    def _log_reply(self, text: str, final, first_ms, total_ms: float) -> None:
        log.info("CHAT <- %r", text.strip())
        usage = getattr(final, "usage", None)
        if usage is None:
            return
        self.calls += 1
        cost = call_cost(self.cfg.model, usage)
        if cost is not None:
            self.spent += cost
        cached = ((getattr(usage, "cache_read_input_tokens", 0) or 0)
                  + (getattr(usage, "cache_creation_input_tokens", 0) or 0))
        log.info("CHAT usage: %d in%s / %d out tokens, stop=%s, first token %s, "
                 "%.0f ms total, %s (session: %d calls, $%.4f)",
                 usage.input_tokens, f" (+{cached} cached)" if cached else "",
                 usage.output_tokens, getattr(final, "stop_reason", "?"),
                 f"{first_ms:.0f} ms" if first_ms is not None else "n/a", total_ms,
                 f"${cost:.5f}" if cost is not None else "cost unknown",
                 self.calls, self.spent)

    def _trim(self) -> None:
        """Bound the context: cap turns, and keep only the newest image.

        Images dominate the token count here, so keeping one holds cost and
        latency roughly flat however long the night runs.
        """
        msgs = self.conversation.messages
        limit = max(2, self.cfg.history_turns * 2)
        if len(msgs) > limit:
            msgs = msgs[-limit:]

        # A tail slice can land on an assistant turn. The API requires the
        # first message to be a user turn and returns 400 otherwise — which
        # would take the prop down mid-party, after enough turns to look random.
        while msgs and msgs[0]["role"] != "user":
            msgs.pop(0)

        seen_image = False
        for msg in reversed(msgs):
            if not isinstance(msg.get("content"), list):
                continue
            kept = []
            for block in msg["content"]:
                if block.get("type") == "image":
                    if seen_image:
                        continue
                    seen_image = True
                kept.append(block)
            msg["content"] = kept

        self.conversation.messages = msgs
