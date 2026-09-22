"""Conversions between our Question/Answer types and the two wire formats in the wild:

- TypeSafe System One (`POST /v1/systemone`), also spoken by Kev and Laya:
    question types noul/choice/score; answers {noul}, {choice, probabilities, confidence}, {score, probabilities, confidence, legend}
- Vercel AI SDK evaluation model (`POST /v4/ai/evaluation-model`):
    question types boolean/choice/score; answers {probability}, {choice, probabilities}, {score, probabilities}
"""

from __future__ import annotations

from typing import Any

from .._types import Answer, Choice, ChoiceAnswer, Noul, NoulAnswer, Question, Score, ScoreAnswer


def questions_to_typesafe(questions: dict[str, Question]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for qid, q in questions.items():
        d: dict[str, Any] = {"type": q.type, "instructions": q.instructions}
        if isinstance(q, Noul):
            if q.criteria:
                d["criteria"] = q.criteria
        else:
            d["criteria"] = q.criteria
        out[qid] = d
    return out


def questions_to_vercel(questions: dict[str, Question]) -> dict[str, Any]:
    out = questions_to_typesafe(questions)
    for d in out.values():
        if d["type"] == "noul":
            d["type"] = "boolean"
    return out


def answers_from_typesafe(raw: dict[str, Any], questions: dict[str, Question]) -> dict[str, Answer]:
    out: dict[str, Answer] = {}
    for qid, q in questions.items():
        a = raw.get(qid)
        if a is None:
            continue
        out[qid] = _one(a, q, style="typesafe")
    return out


def answers_from_vercel(
    raw: dict[str, Any], questions: dict[str, Question], confidence: dict[str, float] | None = None
) -> dict[str, Answer]:
    out: dict[str, Answer] = {}
    for qid, q in questions.items():
        a = raw.get(qid)
        if a is None:
            continue
        ans = _one(a, q, style="vercel")
        if confidence and qid in confidence and hasattr(ans, "confidence"):
            ans.confidence = confidence[qid]  # type: ignore[union-attr]
        out[qid] = ans
    return out


def _one(a: dict[str, Any], q: Question, style: str) -> Answer:
    if isinstance(q, Noul):
        p = a.get("noul", a.get("probability", a.get("p")))
        if p is None and "probabilities" in a:  # some emulators return {true: p, false: 1-p}
            p = a["probabilities"].get("true", a["probabilities"].get(True))
        return NoulAnswer(probability=float(_clamp(p)))
    if isinstance(q, Choice):
        probs = {str(k): float(v) for k, v in (a.get("probabilities") or {}).items()}
        for k in q.criteria:
            probs.setdefault(k, 0.0)
        choice = a.get("choice") or (max(probs, key=probs.get) if probs else next(iter(q.criteria)))
        return ChoiceAnswer(choice=str(choice), probabilities=probs, confidence=a.get("confidence"))
    if isinstance(q, Score):
        raw_probs = a.get("probabilities") or {}
        probs = {int(k): float(v) for k, v in raw_probs.items()}
        for i in range(len(q.criteria)):
            probs.setdefault(i, 0.0)
        score = a.get("score")
        if score is None:
            score = sum(i * p for i, p in probs.items())
        return ScoreAnswer(
            score=float(score),
            probabilities=probs,
            confidence=a.get("confidence"),
            levels=[str(c) if not isinstance(c, str) else c for c in q.criteria],
        )
    raise TypeError(f"unknown question type {type(q)}")


def _clamp(p: Any) -> float:
    try:
        v = float(p)
    except (TypeError, ValueError):
        return 0.5
    return min(1.0, max(0.0, v))
