"""Wake-phrase matching on the STT transcript ("Hello Bones")."""

from __future__ import annotations

import re

# Whisper's usual mishearings, per word of the phrase.
_ALIASES = {
    "hello": ("hello", "hallo", "hullo", "hell o"),
    "hey": ("hey", "hay", "hi", "ay", "a"),
    "bones": ("bones", "bone", "boans"),
}


def _word(word: str) -> str:
    return "(?:" + "|".join(map(re.escape, _ALIASES.get(word, (word,)))) + ")"


def strip_wake_phrase(text: str, phrase: str) -> str | None:
    """None if `text` doesn't contain the wake phrase, else whatever followed it
    ("Hello Bones, nice head" -> "nice head"; "Hello Bones." -> "")."""
    words = re.findall(r"[a-z0-9']+", phrase.lower())
    if not words:
        return text
    pattern = r"\b" + r"\W+".join(_word(w) for w in words) + r"\b"
    match = re.search(pattern, text, re.IGNORECASE)
    if match is None:
        return None
    return text[match.end():].lstrip(" \t,.!?;:-").rstrip()
