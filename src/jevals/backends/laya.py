"""Laya, in-process on Apple Silicon. `pip install "jevals[laya]"`.

Laya's `predict(state, questions)` takes the TypeSafe question shape and returns
`{"answers": {...}}` in the same answer shape, so we reuse the wire converters.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from .._types import BackendResponse, Question, Usage
from ._base import Backend, BackendError
from ._wire import answers_from_typesafe, questions_to_typesafe


class LayaBackend(Backend):
    name = "laya"
    price_per_m_input = 0.0
    price_per_m_output = 0.0

    def __init__(self, model: str | None = None, dtype: str = "float16", **load_kw: Any):
        try:
            import laya_mlx as laya  # type: ignore
        except ImportError as e:
            raise BackendError(
                'laya-mlx is not installed: pip install "jevals[laya]" (Apple Silicon only)'
            ) from e
        self.model_id = model or os.environ.get("JEVALS_LAYA_MODEL", "aac6fef/laya-mlx")
        self._agent = laya.load(self.model_id, dtype=dtype, **load_kw)

    async def evaluate(self, state: Any, questions: dict[str, Question]) -> BackendResponse:
        qs = questions_to_typesafe(questions)
        for q in qs.values():  # laya wants list or dict criteria; instructions must be text
            if not isinstance(q.get("instructions"), str):
                q["instructions"] = str(q.get("instructions"))
        result = await asyncio.to_thread(self._agent.predict, state, qs)
        answers = result.get("answers", result) if isinstance(result, dict) else {}
        usage_raw = result.get("usage", {}) if isinstance(result, dict) else {}
        return BackendResponse(
            answers=answers_from_typesafe(answers, questions),
            usage=Usage(input_tokens=int(usage_raw.get("input_tokens") or 0), backend=self.name),
            raw=result,
        )
