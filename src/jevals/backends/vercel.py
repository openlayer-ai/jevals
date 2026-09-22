"""Jev through Vercel AI Gateway.

Same HTTP call the AI SDK's `experimental_evaluate` makes (read out of @ai-sdk/gateway):
POST https://ai-gateway.vercel.sh/v4/ai/evaluation-model
headers: Authorization: Bearer <AI_GATEWAY_API_KEY | VERCEL_OIDC_TOKEN>
         ai-gateway-protocol-version: 0.0.1
         ai-gateway-auth-method: api-key | oidc
         ai-evaluation-model-specification-version: 4
         ai-model-id: typesafe-ai/jev
body:    {"state": ..., "questions": {id: {type: boolean|choice|score, instructions, criteria}}, "providerOptions": {...}}
"""

from __future__ import annotations

import os
from typing import Any

from .._types import BackendResponse, Question, Usage
from ._base import Backend, BackendError, http_error
from ._wire import answers_from_vercel, questions_to_vercel
from .typesafe import JEV_PRICE_PER_M_INPUT


class VercelBackend(Backend):
    name = "vercel"
    price_per_m_input = JEV_PRICE_PER_M_INPUT
    price_per_m_output = 0.0

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 15.0,  # Jev answers in ~300ms; a hung connection should fail fast and retry
        zero_data_retention: bool = True,
        team: str | None = None,
    ):
        self.api_key = api_key or os.environ.get("AI_GATEWAY_API_KEY")
        self.auth_method = "api-key"
        if not self.api_key and os.environ.get("VERCEL_OIDC_TOKEN"):
            self.api_key = os.environ["VERCEL_OIDC_TOKEN"]
            self.auth_method = "oidc"
        if not self.api_key:
            raise BackendError("AI_GATEWAY_API_KEY (or VERCEL_OIDC_TOKEN) is not set")
        self.model = model or os.environ.get("JEVALS_VERCEL_MODEL", "typesafe-ai/jev")
        self.base_url = (
            base_url or os.environ.get("AI_GATEWAY_BASE_URL") or "https://ai-gateway.vercel.sh/v4/ai"
        ).rstrip("/")
        self.zero_data_retention = zero_data_retention
        self.team = team or os.environ.get("VERCEL_TEAM")
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        h = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "ai-gateway-protocol-version": "0.0.1",
            "ai-gateway-auth-method": self.auth_method,
            "ai-evaluation-model-specification-version": "4",
            "ai-model-id": self.model,
            "user-agent": "jevals",
        }
        if self.team:
            h["ai-gateway-team"] = self.team
        return h

    async def evaluate(self, state: Any, questions: dict[str, Question]) -> BackendResponse:
        body: dict[str, Any] = {"state": state, "questions": questions_to_vercel(questions)}
        if self.zero_data_retention:
            body["providerOptions"] = {"gateway": {"zeroDataRetention": True}}
        r = await self._client().post(f"{self.base_url}/evaluation-model", json=body, headers=self._headers())
        if r.status_code >= 400:
            raise http_error("vercel gateway", r)
        data = r.json()
        conf = ((data.get("providerMetadata") or {}).get("typesafe") or {}).get("confidence")
        usage_raw = data.get("usage") or {}
        return BackendResponse(
            answers=answers_from_vercel(data.get("answers") or {}, questions, conf),
            usage=Usage(
                input_tokens=int(usage_raw.get("inputTokens") or 0),
                output_tokens=int(usage_raw.get("outputTokens") or 0),
                backend=self.name,
            ),
            raw=data,
        )
