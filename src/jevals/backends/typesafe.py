"""TypeSafe System One wire: Jev direct, and anything compatible (Kev, self-hosted).

POST {base_url}/v1/systemone
{"state": ..., "model": "...", "questions": {id: {type, instructions, criteria}}}
"""

from __future__ import annotations

import os
from typing import Any

from .._types import BackendResponse, Question, Usage
from ._base import Backend, BackendError, http_error
from ._wire import answers_from_typesafe, questions_to_typesafe

JEV_PRICE_PER_M_INPUT = 0.042


class TypeSafeBackend(Backend):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        name: str | None = None,
        timeout: float = 15.0,  # Jev answers in ~300ms; a hung connection should fail fast and retry
        zero_data_retention: bool = True,
    ):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        self.base_url = (base_url or os.environ.get("TYPESAFE_BASE_URL") or "https://api.typesafe.ai").rstrip(
            "/"
        )
        self.model = model or os.environ.get("TYPESAFE_MODEL")
        self.name = name or "jev"
        self.timeout = timeout
        self.zero_data_retention = zero_data_retention
        self.price_per_m_input = JEV_PRICE_PER_M_INPUT if self.name == "jev" else 0.0
        self.price_per_m_output = 0.0
        if not self.api_key:
            raise BackendError("TYPESAFE_API_KEY is not set")
        self.timeout = timeout

    async def evaluate(self, state: Any, questions: dict[str, Question]) -> BackendResponse:
        body: dict[str, Any] = {"state": state, "questions": questions_to_typesafe(questions)}
        if self.model:
            body["model"] = self.model
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if self.zero_data_retention:
            headers["X-Data-Retention"] = "zero"
        r = await self._client().post(f"{self.base_url}/v1/systemone", json=body, headers=headers)
        if r.status_code >= 400:
            raise http_error(self.name, r)
        data = r.json()
        usage_raw = data.get("usage") or {}
        return BackendResponse(
            answers=answers_from_typesafe(data.get("answers") or {}, questions),
            usage=Usage(
                input_tokens=int(usage_raw.get("input_tokens") or 0),
                output_tokens=int(usage_raw.get("output_tokens") or 0),
                latency_ms=float(data.get("latency_ms") or 0.0),
                backend=self.name,
            ),
            raw=data,
        )
