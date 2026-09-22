"""Deterministic text helpers. No models."""

from __future__ import annotations

import re

_ABBREV = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "vs",
    "etc",
    "e.g",
    "i.e",
    "inc",
    "ltd",
    "co",
    "no",
    "fig",
    "approx",
}
_SENT_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def split_sentences(text: str, min_chars: int = 12) -> list[str]:
    """Split text into sentences. Conservative: merges abbreviation splits and
    drops fragments shorter than `min_chars` into their neighbor."""
    if not text:
        return []
    # bullets and blank-line paragraphs first: list items often lack terminal punctuation
    blocks = [
        b.strip() for b in re.split(r"\n\s*[-*•]\s+|\n\s*\d+[.)]\s+|\n{2,}", "\n" + text.strip()) if b.strip()
    ]
    out: list[str] = []
    for block in blocks:
        block = re.sub(r"\s+", " ", block)
        parts = _SENT_END.split(block)
        merged: list[str] = []
        for p in parts:
            if merged:
                prev = merged[-1]
                last_word = prev.rstrip(".").split()[-1].lower() if prev.strip() else ""
                if last_word in _ABBREV or len(prev) < min_chars:
                    merged[-1] = prev + " " + p
                    continue
            merged.append(p)
        out.extend(m.strip() for m in merged if m.strip())
    return out or [text.strip()]


def truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0
