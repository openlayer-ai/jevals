"""The Eval base class and Result.

An eval is three functions of a sample: state(), questions(), reduce().
The runner merges states and packs questions across evals into one request.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from ._sample import Sample
from ._types import Answer, Choice, ChoiceAnswer, Noul, NoulAnswer, Question, Score, ScoreAnswer, Usage

Action = Literal["allow", "block", "escalate", "modify"]


class Result(BaseModel):
    """What an eval returns. `score` is 0..1 where higher is better, when it applies."""

    name: str = ""
    score: float | None = None
    passed: bool | None = None
    answer: str | None = None  # for choice-shaped evals
    probability: float | None = None  # the headline probability, when there is one
    detail: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)
    answers: dict[str, Answer] = Field(default_factory=dict)
    action: Action | None = None  # filled by gates
    value: Any = None  # modified payload, for action == "modify"
    skipped: bool = False
    error: str | None = None
    usage: Usage = Field(default_factory=Usage)

    def __bool__(self) -> bool:
        return bool(self.passed) if self.passed is not None else (self.score or 0) >= 0.5

    def __str__(self) -> str:
        bits = [self.name]
        if self.answer is not None:
            bits.append(self.answer)
        elif self.score is not None:
            bits.append(f"{self.score:.2f}")
        if self.passed is not None:
            bits.append("✓" if self.passed else "✗")
        if self.probability is not None:
            bits.append(f"p={self.probability:.2f}")
        if self.detail:
            bits.append(self.detail)
        return "  ".join(bits)


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


class Eval:
    """Subclass and implement state(), questions(), reduce().

    Class attributes:
      name        snake_case identifier; defaults to the class name
      category    "agent" | "security" | "quality" | anything
      requires    sample fields that must be present; missing -> Result(skipped=True)
      threshold   default pass threshold for score-shaped results
    """

    name: ClassVar[str] = ""
    category: ClassVar[str] = "custom"
    description: ClassVar[str] = ""
    requires: ClassVar[tuple[str, ...]] = ()
    threshold: ClassVar[float] = 0.5

    # gate knobs, settable per instance
    block_below: float | None = None
    escalate_below: float | None = None
    block_above: float | None = None
    escalate_above: float | None = None

    def __init__(self, **kw: Any):
        for k, v in kw.items():
            setattr(self, k, v)
        if not self.name:
            type(self).name = _snake(type(self).__name__)

    # ------------------------------------------------------------------ to implement

    def state(self, s: Sample) -> dict[str, Any]:
        """What the model looks at. Deterministic. Keys are merged with other evals' states."""
        return {}

    def questions(self, s: Sample) -> dict[str, Question]:
        raise NotImplementedError

    def reduce(self, answers: dict[str, Answer], s: Sample) -> Result:
        raise NotImplementedError

    # ------------------------------------------------------------------ optional hooks

    def pre(self, s: Sample) -> Result | None:
        """Deterministic pre-check. Return a Result to short-circuit (no questions asked)."""
        return None

    def decide(self, r: Result, s: Sample) -> Action | None:
        """Gate policy. Default: use block_below / escalate_below on the headline value."""
        v = r.probability if r.probability is not None else r.score
        if v is None:
            return None
        if self.block_above is not None and v >= self.block_above:
            return "block"
        if self.block_below is not None and v < self.block_below:
            return "block"
        if self.escalate_above is not None and v >= self.escalate_above:
            return "escalate"
        if self.escalate_below is not None and v < self.escalate_below:
            return "escalate"
        return None

    # ------------------------------------------------------------------ plumbing

    def applicable(self, s: Sample) -> tuple[bool, str]:
        missing = [k for k in self.requires if not s.has(k)]
        if missing:
            return False, f"missing {', '.join(missing)}"
        return True, ""

    @property
    def key(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description or (self.__doc__ or "").strip().split("\n")[0],
            "requires": list(self.requires),
        }


# ---------------------------------------------------------------------- helpers for one-question evals


class NoulEval(Eval):
    """One yes/no question. Subclass with `instructions` and optionally `criteria`.

    `positive_is_good`: if False (e.g. "does this contain prompt injection?"),
    score = 1 - p and passed = p < threshold.
    """

    instructions: ClassVar[str] = ""
    criteria: ClassVar[dict[str, str] | None] = None
    positive_is_good: ClassVar[bool] = True

    def questions(self, s: Sample) -> dict[str, Question]:
        return {"q": Noul(self.instructions, criteria=self.criteria)}

    def reduce(self, answers: dict[str, Answer], s: Sample) -> Result:
        a = answers["q"]
        assert isinstance(a, NoulAnswer)
        p = a.probability
        if self.positive_is_good:
            return Result(score=p, passed=p >= self.threshold, probability=p)
        return Result(score=1 - p, passed=p < self.threshold, probability=p)

    def decide(self, r: Result, s: Sample) -> Action | None:
        # for "bad thing present" evals, block_below is written in terms of the *safe* score
        # so `PromptInjection(block_below=0.5)` blocks when p(injection) > 0.5. Use r.score.
        v = r.score
        if v is None:
            return None
        if self.block_below is not None and v < self.block_below:
            return "block"
        if self.escalate_below is not None and v < self.escalate_below:
            return "escalate"
        return None


class ScoreEval(Eval):
    """One rubric question. Subclass with `instructions` and `levels` (lowest first)."""

    instructions: ClassVar[str] = ""
    levels: ClassVar[list[str]] = []

    def __init__(self, levels: list[str] | None = None, **kw: Any):
        super().__init__(**kw)
        if levels:
            self.levels = levels  # type: ignore[misc]

    def questions(self, s: Sample) -> dict[str, Question]:
        return {"q": Score(self.instructions, criteria=list(self.levels))}

    def reduce(self, answers: dict[str, Answer], s: Sample) -> Result:
        a = answers["q"]
        assert isinstance(a, ScoreAnswer)
        n = len(self.levels)
        norm = a.score / (n - 1) if n > 1 else 0.0
        top = a.label or (self.levels[a.level] if 0 <= a.level < n else "")
        probs = ", ".join(
            f"{self.levels[i]}: {p:.2f}"
            for i, p in sorted(a.probabilities.items(), key=lambda kv: -kv[1])[:3]
        )
        return Result(
            score=norm,
            passed=norm >= self.threshold,
            probability=None,
            detail=f"{a.score:.1f} / {n - 1}  {{{probs}}}",
            evidence={"level": a.level, "label": top, "probabilities": a.probabilities},
        )


class ChoiceEval(Eval):
    """One choice question. Subclass with `instructions`, `options` and `good` (the acceptable option keys)."""

    instructions: ClassVar[str] = ""
    options: ClassVar[dict[str, str]] = {}
    good: ClassVar[tuple[str, ...]] = ()

    def __init__(self, options: dict[str, str] | None = None, good: tuple[str, ...] | None = None, **kw: Any):
        super().__init__(**kw)
        if options:
            self.options = options  # type: ignore[misc]
        if good is not None:
            self.good = tuple(good)  # type: ignore[misc]
        elif options and not self.good:
            self.good = (next(iter(options)),)  # type: ignore[misc]

    def questions(self, s: Sample) -> dict[str, Question]:
        return {"q": Choice(self.instructions, criteria=dict(self.options))}

    def reduce(self, answers: dict[str, Answer], s: Sample) -> Result:
        a = answers["q"]
        assert isinstance(a, ChoiceAnswer)
        p_good = sum(a.p(o) for o in self.good) if self.good else a.probability
        return Result(
            score=p_good,
            passed=a.choice in self.good if self.good else None,
            answer=a.choice,
            probability=a.probability,
            detail=f"conf={a.confidence:.2f}" if a.confidence is not None else "",
            evidence={"probabilities": a.probabilities, "confidence": a.confidence},
        )
