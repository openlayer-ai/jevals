from __future__ import annotations

from typing import Any

from .._eval import Eval, NoulEval, Result, ScoreEval
from .._sample import Sample
from .._text import mean, split_sentences, truncate
from .._types import Answer, Choice, ChoiceAnswer, Noul, Question, Score, ScoreAnswer

# --------------------------------------------------------------------------- RAG


class Faithfulness(Eval):
    """Every statement in the answer is supported by the retrieved context. Ragas faithfulness in one request."""

    name = "faithfulness"
    category = "quality"
    requires = ("contexts", "output")

    def __init__(self, threshold: float = 0.7, **kw: Any):
        super().__init__(**kw)
        self.threshold = threshold  # type: ignore[misc]

    def _claims(self, s: Sample) -> list[str]:
        return split_sentences(s.output)[:25]

    def state(self, s: Sample) -> dict[str, Any]:
        return {"contexts": [truncate(c, 3000) for c in s.contexts], "statements": self._claims(s)}

    def questions(self, s: Sample) -> dict[str, Question]:
        return {
            f"s{i}": Noul(
                f"Can statements[{i}] be inferred from contexts alone?",
                criteria={
                    "true": "Directly stated or a straightforward inference from contexts",
                    "false": "Requires facts not present in contexts, or contradicts them",
                },
            )
            for i in range(len(self._claims(s)))
        }

    def reduce(self, answers: dict[str, Answer], s: Sample) -> Result:
        claims = self._claims(s)
        ps = [answers[f"s{i}"].probability for i in range(len(claims)) if f"s{i}" in answers]  # type: ignore[union-attr]
        if not ps:
            return Result(score=None, detail="no statements")
        ok = [p >= 0.5 for p in ps]
        score = mean(ok)
        return Result(
            score=score,
            passed=score >= self.threshold,
            detail=f"{sum(ok)}/{len(ok)} statements supported",
            evidence={"per_statement": ps, "statements": claims},
        )


class AnswerRelevancy(Eval):
    """Does the answer address the question, directly and without hedging? Ragas answer relevancy without the reverse-question trick."""

    name = "answer_relevancy"
    category = "quality"
    requires = ("input", "output")

    def state(self, s: Sample) -> dict[str, Any]:
        return {"question": truncate(s.input, 2000), "answer": truncate(s.output, 4000)}

    def questions(self, s: Sample) -> dict[str, Question]:
        return {
            "relevance": Score(
                "How well does answer address question?",
                criteria=[
                    "Off-topic or answers a different question",
                    "Touches the topic but misses what was asked",
                    "Addresses the question with digressions or missing parts",
                    "Addresses the question directly and completely",
                ],
            ),
            "noncommittal": Noul(
                "Is answer noncommittal or evasive (e.g. 'I don't know', 'it depends', asks a question back without answering)?",
            ),
        }

    def reduce(self, answers: dict[str, Answer], s: Sample) -> Result:
        rel = answers["relevance"]
        assert isinstance(rel, ScoreAnswer)
        nc = answers["noncommittal"].probability  # type: ignore[union-attr]
        score = (rel.score / 3) * (1 - nc)
        return Result(
            score=score,
            passed=score >= self.threshold,
            detail=f"relevance {rel.score:.1f}/3, p(noncommittal)={nc:.2f}",
            evidence={"relevance": rel.score, "noncommittal": nc},
        )


class ContextPrecision(Eval):
    """Are the retrieved chunks relevant, and are the relevant ones ranked first? Average precision over one question per chunk."""

    name = "context_precision"
    category = "quality"
    requires = ("input", "contexts")

    def state(self, s: Sample) -> dict[str, Any]:
        st: dict[str, Any] = {
            "question": truncate(s.input, 2000),
            "contexts": [truncate(c, 2500) for c in s.contexts[:20]],
        }
        if s.expected:
            st["reference_answer"] = truncate(s.expected, 2000)
        return st

    def questions(self, s: Sample) -> dict[str, Question]:
        target = "reference_answer" if s.expected else "question"
        return {
            f"chunk{i}": Noul(
                f"Is contexts[{i}] useful for answering the {target}?",
                criteria={
                    "true": "Contains information needed for the answer",
                    "false": "Unrelated, or related but not needed",
                },
            )
            for i in range(min(len(s.contexts), 20))
        }

    def reduce(self, answers: dict[str, Answer], s: Sample) -> Result:
        n = min(len(s.contexts), 20)
        rel = [answers[f"chunk{i}"].probability >= 0.5 for i in range(n) if f"chunk{i}" in answers]  # type: ignore[union-attr]
        if not rel:
            return Result(score=None, detail="no contexts")
        hits, ap = 0, 0.0
        for k, r in enumerate(rel, start=1):
            if r:
                hits += 1
                ap += hits / k
        score = ap / hits if hits else 0.0
        irrelevant = [i for i, r in enumerate(rel) if not r]
        detail = f"AP over {n} chunks" + (f"; irrelevant: {irrelevant}" if irrelevant else "")
        return Result(score=score, passed=score >= self.threshold, detail=detail, evidence={"relevant": rel})


class ContextRecall(Eval):
    """Does the retrieved context cover the reference answer? One question per reference sentence."""

    name = "context_recall"
    category = "quality"
    requires = ("contexts", "expected")

    def _claims(self, s: Sample) -> list[str]:
        return split_sentences(s.expected or "")[:25]

    def state(self, s: Sample) -> dict[str, Any]:
        return {"contexts": [truncate(c, 3000) for c in s.contexts], "reference_statements": self._claims(s)}

    def questions(self, s: Sample) -> dict[str, Question]:
        return {
            f"r{i}": Noul(f"Can reference_statements[{i}] be attributed to contexts?")
            for i in range(len(self._claims(s)))
        }

    def reduce(self, answers: dict[str, Answer], s: Sample) -> Result:
        n = len(self._claims(s))
        ok = [answers[f"r{i}"].probability >= 0.5 for i in range(n) if f"r{i}" in answers]  # type: ignore[union-attr]
        if not ok:
            return Result(score=None, detail="no reference statements")
        score = mean(ok)
        return Result(
            score=score,
            passed=score >= self.threshold,
            detail=f"{sum(ok)}/{len(ok)} reference statements covered",
        )


class Hallucination(Faithfulness):
    """Claims in the output that are not in the context. Inverse of faithfulness, kept for people who think in that direction."""

    name = "hallucination"

    def reduce(self, answers: dict[str, Answer], s: Sample) -> Result:
        r = super().reduce(answers, s)
        if r.score is None:
            return r
        halluc = 1 - r.score
        n = len(r.evidence.get("per_statement", []))
        return Result(
            score=r.score,
            passed=halluc <= (1 - self.threshold),
            probability=halluc,
            detail=f"{round(halluc * n)}/{n} statements unsupported",
            evidence=r.evidence,
        )


# --------------------------------------------------------------------------- general


class Correctness(ScoreEval):
    """Agreement with the expected answer."""

    name = "correctness"
    category = "quality"
    requires = ("output", "expected")
    instructions = "How well does answer agree with reference on the facts that matter?"
    levels = [
        "Contradicts or misses the reference",
        "Partially correct with significant errors or omissions",
        "Mostly correct with minor differences",
        "Equivalent to the reference in substance",
    ]
    threshold = 0.66

    def state(self, s: Sample) -> dict[str, Any]:
        return {
            "question": truncate(s.input, 1500),
            "answer": truncate(s.output, 4000),
            "reference": truncate(s.expected or "", 4000),
        }


class Completeness(ScoreEval):
    """Does the answer cover everything the question asked?"""

    name = "completeness"
    category = "quality"
    requires = ("input", "output")
    instructions = "How completely does answer cover every part of question?"
    levels = [
        "Ignores most of the question",
        "Covers some parts",
        "Covers most parts",
        "Covers every part of the question",
    ]
    threshold = 0.66

    def state(self, s: Sample) -> dict[str, Any]:
        return {"question": truncate(s.input, 2000), "answer": truncate(s.output, 4000)}


class Coherence(ScoreEval):
    """Logical flow and structure."""

    name = "coherence"
    category = "quality"
    requires = ("output",)
    instructions = "How coherent is text: logical order, consistent claims, readable structure?"
    levels = [
        "Disorganized or self-contradicting",
        "Hard to follow in places",
        "Clear with minor rough edges",
        "Well organized and consistent throughout",
    ]
    threshold = 0.5

    def state(self, s: Sample) -> dict[str, Any]:
        return {"text": truncate(s.output, 6000)}


class InstructionFollowing(NoulEval):
    """Did the output follow the explicit instructions (format, length, constraints) in the prompt?"""

    name = "instruction_following"
    category = "quality"
    requires = ("output",)
    instructions = "Does output follow every explicit instruction in system_prompt and request: format, length, language, constraints, things to avoid?"

    def state(self, s: Sample) -> dict[str, Any]:
        return {
            "system_prompt": truncate(s.system_prompt, 3000),
            "request": truncate(s.input, 2000),
            "output": truncate(s.output, 4000),
        }


class Refusal(Eval):
    """Did the model refuse, and was the refusal warranted? Catches over-refusal of safe requests."""

    name = "refusal"
    category = "quality"
    requires = ("input", "output")

    def state(self, s: Sample) -> dict[str, Any]:
        return {"request": truncate(s.input, 2000), "output": truncate(s.output, 3000)}

    def questions(self, s: Sample) -> dict[str, Question]:
        return {
            "kind": Choice(
                "How did the assistant respond to request?",
                criteria={
                    "answered": "Provided what was asked",
                    "warranted_refusal": "Declined, and the request was harmful, illegal, or outside a stated policy",
                    "over_refusal": "Declined or heavily hedged a request that was safe and reasonable to fulfill",
                    "partial": "Answered part of it and declined part of it",
                },
            )
        }

    def reduce(self, answers: dict[str, Answer], s: Sample) -> Result:
        a = answers["kind"]
        assert isinstance(a, ChoiceAnswer)
        bad = a.p("over_refusal")
        return Result(
            score=1 - bad,
            passed=a.choice != "over_refusal",
            answer=a.choice,
            probability=a.probability,
            evidence={"probabilities": a.probabilities},
        )


class CustomRubric(ScoreEval):
    """Your rubric. `CustomRubric("How well does the reply match our brand voice?", levels=[...])`."""

    name = "custom_rubric"
    category = "quality"
    requires = ("output",)

    def __init__(
        self,
        instructions: str,
        levels: list[str],
        name: str | None = None,
        threshold: float = 0.5,
        fields: tuple[str, ...] = ("input", "output"),
        **kw: Any,
    ):
        super().__init__(levels=levels, **kw)
        self.instructions = instructions  # type: ignore[misc]
        self.threshold = threshold  # type: ignore[misc]
        self.fields = fields
        if name:
            self.name = name  # type: ignore[misc]

    def state(self, s: Sample) -> dict[str, Any]:
        return {f: truncate(str(getattr(s, f) or ""), 4000) for f in self.fields}
