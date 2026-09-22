"""Deterministic backend for tests. No network.

MockBackend()                                  # noul 0.5, uniform choice/score
MockBackend(default_noul=0.9)                  # every yes/no is 0.9
MockBackend(answers={"grounded.c0": 0.2})      # per packed question id
MockBackend(fn=lambda qid, q, state: 0.7)      # your rule; return a float, a label, or a dict of probabilities
"""

from __future__ import annotations

import fnmatch
import json
from collections.abc import Callable
from typing import Any

from .._types import (
    Answer,
    BackendResponse,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
    Usage,
)
from ._base import Backend


class MockBackend(Backend):
    name = "mock"
    price_per_m_input = 0.0
    price_per_m_output = 0.0

    def __init__(
        self,
        answers: dict[str, Any] | None = None,
        default_noul: float = 0.5,
        fn: Callable[[str, Question, Any], Any] | None = None,
    ):
        self.answers = answers or {}
        self.default_noul = default_noul
        self.fn = fn
        self.calls: list[dict[str, Any]] = []

    def _lookup(self, qid: str) -> Any:
        if qid in self.answers:
            return self.answers[qid]
        for pat, v in self.answers.items():
            if fnmatch.fnmatch(qid, pat):
                return v
        # also match on the un-namespaced tail ("c0" for "grounded.c0")
        tail = qid.split(".", 1)[-1]
        if tail in self.answers:
            return self.answers[tail]
        return None

    async def evaluate(self, state: Any, questions: dict[str, Question]) -> BackendResponse:
        self.calls.append({"state": state, "questions": {k: q.model_dump() for k, q in questions.items()}})
        out: dict[str, Answer] = {}
        for qid, q in questions.items():
            v = self.fn(qid, q, state) if self.fn else self._lookup(qid)
            out[qid] = _make(v, q, self.default_noul)
        tokens = len(json.dumps(state, default=str)) // 4 + sum(
            len(json.dumps(q.model_dump(), default=str)) // 4 for q in questions.values()
        )
        return BackendResponse(answers=out, usage=Usage(input_tokens=tokens, backend=self.name, cost_usd=0.0))


def _make(v: Any, q: Question, default_noul: float) -> Answer:
    if isinstance(q, Noul):
        if v is None:
            v = default_noul
        if isinstance(v, bool):
            v = 0.95 if v else 0.05
        return NoulAnswer(probability=float(v))
    if isinstance(q, Choice):
        keys = list(q.criteria)
        if isinstance(v, dict):
            probs = {k: float(v.get(k, 0.0)) for k in keys}
        elif isinstance(v, str) and v in keys:
            probs = {k: (0.9 if k == v else 0.1 / max(1, len(keys) - 1)) for k in keys}
        else:
            probs = {k: 1.0 / len(keys) for k in keys}
        s = sum(probs.values()) or 1.0
        probs = {k: p / s for k, p in probs.items()}
        return ChoiceAnswer(choice=max(probs, key=probs.get), probabilities=probs)
    if isinstance(q, Score):
        n = len(q.criteria)
        if isinstance(v, dict):
            probs = {int(k): float(p) for k, p in v.items()}
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            lvl = int(round(float(v)))
            probs = {i: (1.0 if i == lvl else 0.0) for i in range(n)}
        else:
            probs = {i: 1.0 / n for i in range(n)}
        for i in range(n):
            probs.setdefault(i, 0.0)
        s = sum(probs.values()) or 1.0
        probs = {k: p / s for k, p in probs.items()}
        return ScoreAnswer(
            score=sum(i * p for i, p in probs.items()),
            probabilities=probs,
            levels=[str(c) for c in q.criteria],
        )
    raise TypeError(type(q))
