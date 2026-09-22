"""Backends. Anything that answers typed questions about state.

Resolution order when `backend` is not given:
  JEVALS_BACKEND (explicit spec) > TYPESAFE_API_KEY > AI_GATEWAY_API_KEY > KEV_BASE_URL > OPENROUTER_API_KEY / OPENAI_API_KEY
"""

from __future__ import annotations

import os
from typing import Any

from ._base import Backend, BackendError
from .mock import MockBackend

__all__ = ["Backend", "BackendError", "MockBackend", "resolve"]

_cache: dict[str, Backend] = {}


def resolve(spec: str | Backend | None = None) -> Backend:
    """Turn a backend spec into a Backend.

    Specs: "jev" | "typesafe" | "vercel" | "kev" | "kev://host:port" | "laya" | "laya:<hf-id>"
           | "llm:<model>" | "openrouter:<model>" | "mock" | full http(s) URL to a /v1/systemone server
    """
    if isinstance(spec, Backend):
        return spec
    spec = spec or os.environ.get("JEVALS_BACKEND") or _autodetect()
    if spec in _cache:
        return _cache[spec]
    b = _build(spec)
    _cache[spec] = b
    return b


def _autodetect() -> str:
    env = os.environ
    if env.get("TYPESAFE_API_KEY"):
        return "jev"
    if env.get("AI_GATEWAY_API_KEY") or env.get("VERCEL_OIDC_TOKEN"):
        return "vercel"
    if env.get("KEV_BASE_URL"):
        return "kev"
    if env.get("OPENROUTER_API_KEY"):
        return "llm:" + env.get("JEVALS_LLM_MODEL", "openai/gpt-4.1-mini")
    if env.get("OPENAI_API_KEY"):
        return "llm:" + env.get("JEVALS_LLM_MODEL", "gpt-4.1-mini")
    raise BackendError(
        "No backend configured. Set one of TYPESAFE_API_KEY, AI_GATEWAY_API_KEY, KEV_BASE_URL, "
        "OPENROUTER_API_KEY, or pass backend=... (e.g. backend='mock')."
    )


def _build(spec: str) -> Backend:
    name, _, rest = spec.partition(":")
    name = name.lower()
    if name in ("jev", "typesafe"):
        from .typesafe import TypeSafeBackend

        return TypeSafeBackend(model=rest or None)
    if name == "vercel":
        from .vercel import VercelBackend

        return VercelBackend(model=rest or None)
    if name == "kev":
        from .typesafe import TypeSafeBackend

        url = rest.lstrip("/") if rest else os.environ.get("KEV_BASE_URL", "localhost:8009")
        if not url.startswith("http"):
            url = "http://" + url
        return TypeSafeBackend(base_url=url, api_key="local", model="kev-latest", name="kev")
    if name in ("http", "https"):
        from .typesafe import TypeSafeBackend

        return TypeSafeBackend(
            base_url=spec, api_key=os.environ.get("TYPESAFE_API_KEY", "local"), name="systemone"
        )
    if name == "laya":
        from .laya import LayaBackend

        return LayaBackend(model=rest or None)
    if name in ("llm", "openrouter", "openai"):
        from .llm import LLMBackend

        return LLMBackend(model=rest or None, provider=name if name != "llm" else None)
    if name == "mock":
        return MockBackend()
    raise BackendError(f"unknown backend spec {spec!r}")


def register(name: str, factory: Any) -> None:
    """Register a custom backend under a spec name."""
    _cache[name] = factory() if callable(factory) and not isinstance(factory, Backend) else factory
