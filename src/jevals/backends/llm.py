"""Emulate noul/choice/score with a chat LLM through an OpenAI-compatible API.

One call per request. The model is asked for probability estimates in JSON,
which we normalize. This is the slow, expensive fallback; probabilities here
are the model's own guesses, not calibrated outputs. Useful when you don't
have a System One backend yet, or as a "thorough" tier for questions that
need reasoning.
"""

from __future__ import annotations

import json
import os
import re
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
from ._base import Backend, BackendError, http_error

_PROVIDERS = {
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
}

SYSTEM = """You are a decision model. You do not write prose. You read STATE and answer each QUESTION independently.
Return one JSON object keyed by question id. For each question:
- type "noul": a number in [0,1], the probability the statement is true.
- type "choice": an object mapping every option key to a probability; they must sum to 1.
- type "score": an object mapping every level index ("0","1",...) to a probability; they must sum to 1.
Base every answer only on STATE. If STATE does not contain enough evidence, spread probability accordingly.
Output JSON only."""


class LLMBackend(Backend):
    price_per_m_input = None

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        temperature: float = 0.0,
    ):
        provider = provider or ("openrouter" if os.environ.get("OPENROUTER_API_KEY") else "openai")
        default_url, key_env = _PROVIDERS.get(provider, _PROVIDERS["openai"])
        self.base_url = (base_url or os.environ.get("JEVALS_LLM_BASE_URL") or default_url).rstrip("/")
        self.api_key = (
            api_key
            or os.environ.get(key_env)
            or os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not self.api_key:
            raise BackendError(f"{key_env} is not set")
        self.model = (
            model
            or os.environ.get("JEVALS_LLM_MODEL")
            or ("openai/gpt-4.1-mini" if provider == "openrouter" else "gpt-4.1-mini")
        )
        self.name = f"llm:{self.model}"
        self.temperature = temperature
        self.timeout = timeout

    def _prompt(self, state: Any, questions: dict[str, Question]) -> str:
        qs = {}
        for qid, q in questions.items():
            d: dict[str, Any] = {"type": q.type, "instructions": q.instructions}
            if isinstance(q, Noul) and q.criteria:
                d["criteria"] = q.criteria
            elif isinstance(q, Choice):
                d["options"] = q.criteria
            elif isinstance(q, Score):
                d["levels"] = {str(i): lvl for i, lvl in enumerate(q.criteria)}
            qs[qid] = d
        return (
            "STATE:\n"
            + json.dumps(state, ensure_ascii=False, default=str)
            + "\n\nQUESTIONS:\n"
            + json.dumps(qs, ensure_ascii=False)
        )

    async def evaluate(self, state: Any, questions: dict[str, Question]) -> BackendResponse:
        body = {
            "model": self.model,
            "temperature": self.temperature,
            # The reply is one small JSON object. Without a cap, providers that
            # pre-authorize the model's full output limit (OpenRouter does) reject
            # the request on a low balance.
            "max_tokens": 256 + 64 * len(questions),
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": self._prompt(state, questions)},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if "openrouter" in self.base_url:
            headers["HTTP-Referer"] = "https://github.com/openlayer-ai/jevals"
            headers["X-Title"] = "jevals"
        r = await self._client().post(f"{self.base_url}/chat/completions", json=body, headers=headers)
        if r.status_code >= 400:
            raise http_error(self.name, r)
        data = r.json()
        text = data["choices"][0]["message"]["content"] or "{}"
        parsed = _parse_json(text)
        usage_raw = data.get("usage") or {}
        return BackendResponse(
            answers={qid: _to_answer(parsed.get(qid), q) for qid, q in questions.items()},
            usage=Usage(
                input_tokens=int(usage_raw.get("prompt_tokens") or 0),
                output_tokens=int(usage_raw.get("completion_tokens") or 0),
                cost_usd=_cost(data),
                backend=self.name,
            ),
            raw=data,
        )


def _cost(data: dict[str, Any]) -> float | None:
    u = data.get("usage") or {}
    if "cost" in u:  # OpenRouter reports this
        try:
            return float(u["cost"])
        except (TypeError, ValueError):
            return None
    return None


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        v = json.loads(text)
        return v if isinstance(v, dict) else {}
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def _norm(d: dict[Any, Any], keys: list[str]) -> dict[str, float]:
    out = {}
    for k in keys:
        v = d.get(k, d.get(str(k), 0.0)) if isinstance(d, dict) else 0.0
        try:
            out[str(k)] = max(0.0, float(v))
        except (TypeError, ValueError):
            out[str(k)] = 0.0
    total = sum(out.values())
    if total <= 0:
        return {k: 1.0 / len(keys) for k in out}
    return {k: v / total for k, v in out.items()}


def _to_answer(v: Any, q: Question) -> Answer:
    if isinstance(q, Noul):
        if isinstance(v, dict):
            v = v.get("probability", v.get("true", v.get("noul", 0.5)))
        if isinstance(v, bool):
            v = 1.0 if v else 0.0
        try:
            p = min(1.0, max(0.0, float(v)))
        except (TypeError, ValueError):
            p = 0.5
        return NoulAnswer(probability=p)
    if isinstance(q, Choice):
        keys = list(q.criteria)
        if isinstance(v, str) and v in keys:  # model returned a bare label
            v = {k: (0.9 if k == v else 0.1 / max(1, len(keys) - 1)) for k in keys}
        probs = _norm(v if isinstance(v, dict) else {}, keys)
        return ChoiceAnswer(choice=max(probs, key=probs.get), probabilities=probs)
    if isinstance(q, Score):
        n = len(q.criteria)
        if isinstance(v, (int, float)) and not isinstance(v, bool):  # bare level
            v = {str(i): (1.0 if i == int(round(v)) else 0.0) for i in range(n)}
        probs_s = _norm(v if isinstance(v, dict) else {}, [str(i) for i in range(n)])
        probs = {int(k): p for k, p in probs_s.items()}
        return ScoreAnswer(
            score=sum(i * p for i, p in probs.items()),
            probabilities=probs,
            levels=[str(c) for c in q.criteria],
        )
    raise TypeError(type(q))
