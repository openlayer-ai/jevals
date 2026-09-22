"""Typed questions and answers. This is the System One contract.

Three question types, three answer types. Everything else in jevals is built
on top of these.
"""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, Field, model_validator

JSONContent = Union[str, dict, list]


# --------------------------------------------------------------------------- questions


class Noul(BaseModel):
    """A yes/no question. The answer is P(true)."""

    type: Literal["noul"] = "noul"
    instructions: JSONContent
    criteria: dict[str, JSONContent | None] | None = None  # {"true": ..., "false": ...}

    def __init__(self, instructions: JSONContent | None = None, /, **kw: Any):
        if instructions is not None:
            kw["instructions"] = instructions
        super().__init__(**kw)


class Choice(BaseModel):
    """Pick one of several named options. The answer is a distribution over them."""

    type: Literal["choice"] = "choice"
    instructions: JSONContent
    criteria: dict[str, JSONContent | None]  # option key -> description

    def __init__(self, instructions: JSONContent | None = None, /, **kw: Any):
        if instructions is not None:
            kw["instructions"] = instructions
        if "options" in kw and "criteria" not in kw:
            kw["criteria"] = kw.pop("options")
        super().__init__(**kw)

    @model_validator(mode="after")
    def _check(self) -> Choice:
        if not 1 <= len(self.criteria) <= 255:
            raise ValueError("choice needs between 1 and 255 options")
        return self


class Score(BaseModel):
    """Grade against an ordered rubric. criteria[0] is the lowest level."""

    type: Literal["score"] = "score"
    instructions: JSONContent
    criteria: list[JSONContent]  # ordered, lowest first

    def __init__(self, instructions: JSONContent | None = None, /, **kw: Any):
        if instructions is not None:
            kw["instructions"] = instructions
        if "levels" in kw and "criteria" not in kw:
            kw["criteria"] = kw.pop("levels")
        super().__init__(**kw)

    @model_validator(mode="after")
    def _check(self) -> Score:
        if not 2 <= len(self.criteria) <= 10:
            raise ValueError("score needs between 2 and 10 levels")
        return self


Question = Union[Noul, Choice, Score]


def question_from_dict(d: dict[str, Any]) -> Question:
    t = d.get("type")
    if t in ("noul", "boolean", "bool"):
        return Noul(**{**d, "type": "noul"})
    if t == "choice":
        return Choice(**d)
    if t == "score":
        return Score(**d)
    raise ValueError(f"unknown question type: {t!r}")


# --------------------------------------------------------------------------- answers


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    probability: float = Field(ge=0.0, le=1.0)

    @property
    def value(self) -> bool:
        return self.probability >= 0.5

    def __float__(self) -> float:
        return self.probability


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float | None = None  # (p_max - 1/K) / (1 - 1/K); None if the backend didn't send one

    @property
    def probability(self) -> float:
        """Probability of the selected option."""
        return self.probabilities.get(self.choice, 0.0)

    def p(self, option: str) -> float:
        return self.probabilities.get(option, 0.0)

    @model_validator(mode="after")
    def _fill_confidence(self) -> ChoiceAnswer:
        if self.confidence is None and self.probabilities:
            k = len(self.probabilities)
            pmax = max(self.probabilities.values())
            self.confidence = (pmax - 1 / k) / (1 - 1 / k) if k > 1 else 1.0
        return self


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float  # expected level, 0-indexed, may be fractional
    probabilities: dict[int, float]  # level index -> probability
    confidence: float | None = None
    levels: list[str] | None = None  # rubric text, if known

    @property
    def n_levels(self) -> int:
        return len(self.levels) if self.levels else (max(self.probabilities) + 1 if self.probabilities else 2)

    @property
    def normalized(self) -> float:
        """Score rescaled to 0..1."""
        n = self.n_levels
        return self.score / (n - 1) if n > 1 else 0.0

    @property
    def level(self) -> int:
        """Most likely level index."""
        return (
            max(self.probabilities, key=self.probabilities.get) if self.probabilities else round(self.score)
        )

    @property
    def label(self) -> str | None:
        return self.levels[self.level] if self.levels and 0 <= self.level < len(self.levels) else None

    @model_validator(mode="after")
    def _fill_confidence(self) -> ScoreAnswer:
        if self.confidence is None and self.probabilities:
            k = len(self.probabilities)
            pmax = max(self.probabilities.values())
            self.confidence = (pmax - 1 / k) / (1 - 1 / k) if k > 1 else 1.0
        return self


Answer = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]


# --------------------------------------------------------------------------- usage


class Usage(BaseModel):
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    latency_ms: float = 0.0
    backend: str | None = None

    def __add__(self, other: Usage) -> Usage:
        cost = None
        if self.cost_usd is not None or other.cost_usd is not None:
            cost = (self.cost_usd or 0.0) + (other.cost_usd or 0.0)
        return Usage(
            requests=self.requests + other.requests,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=cost,
            latency_ms=self.latency_ms + other.latency_ms,
            backend=self.backend or other.backend,
        )

    def __str__(self) -> str:
        parts = [
            f"{self.requests} request{'s' if self.requests != 1 else ''}",
            f"{self.input_tokens:,} tokens",
        ]
        if self.cost_usd is not None:
            parts.append(f"${self.cost_usd:.5f}" if self.cost_usd < 0.01 else f"${self.cost_usd:.2f}")
        parts.append(f"{self.latency_ms / 1000:.2f}s")
        return " · ".join(parts)


class BackendResponse(BaseModel):
    answers: dict[str, Answer]
    usage: Usage = Field(default_factory=Usage)
    raw: Any = None
