from __future__ import annotations

import asyncio
import concurrent.futures
import random
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx

from .._types import BackendResponse, Question, Usage


class BackendError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        return self.status == 429 or (self.status is not None and self.status >= 500)


def http_error(name: str, r: httpx.Response) -> BackendError:
    """BackendError from an httpx response, keeping status and Retry-After."""
    ra = r.headers.get("retry-after")
    try:
        retry_after = float(ra) if ra else None
    except ValueError:
        retry_after = None
    return BackendError(
        f"{name} {r.status_code}: {r.text[:500]}", status=r.status_code, retry_after=retry_after
    )


class Backend(ABC):
    """Answer typed questions about a state. Implement `evaluate`."""

    name: str = "backend"
    price_per_m_input: float | None = None  # USD per million input tokens; None = unknown
    price_per_m_output: float | None = None
    timeout: float = 60.0
    max_retries: int = 4  # on 429 / 5xx / connection errors; backoff 0.5, 1, 2, 4s + jitter

    _http: httpx.AsyncClient | None = None
    _http_loop: asyncio.AbstractEventLoop | None = None

    def _client(self) -> httpx.AsyncClient:
        """An httpx client bound to the current event loop.

        Sync evaluate() calls asyncio.run(), which creates and closes a loop each
        time. A client created under the first loop cannot be used under the
        second, so we keep one per loop and rebuild when the loop changes.
        Within a single loop (a dataset run) connections are still pooled.
        """
        loop = asyncio.get_running_loop()
        if self._http is None or self._http_loop is not loop or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=self.timeout)
            self._http_loop = loop
        return self._http

    @abstractmethod
    async def evaluate(self, state: Any, questions: dict[str, Question]) -> BackendResponse: ...

    async def run(self, state: Any, questions: dict[str, Question]) -> BackendResponse:
        """evaluate() plus retries, timing and cost."""
        t0 = time.perf_counter()
        attempt = 0
        while True:
            try:
                resp = await self.evaluate(state, questions)
                break
            except (BackendError, httpx.TransportError) as e:
                retryable = e.retryable if isinstance(e, BackendError) else True
                if not retryable or attempt >= self.max_retries:
                    raise
                delay = 0.5 * 2**attempt + random.uniform(0, 0.25)
                if isinstance(e, BackendError) and e.retry_after:
                    delay = max(delay, min(e.retry_after, 30.0))
                await asyncio.sleep(delay)
                attempt += 1
        resp.usage.latency_ms = resp.usage.latency_ms or (time.perf_counter() - t0) * 1000
        resp.usage.requests = resp.usage.requests or 1
        resp.usage.backend = resp.usage.backend or self.name
        if resp.usage.cost_usd is None and self.price_per_m_input is not None:
            resp.usage.cost_usd = (
                resp.usage.input_tokens / 1e6 * self.price_per_m_input
                + resp.usage.output_tokens / 1e6 * (self.price_per_m_output or 0.0)
            )
        return resp

    def run_sync(self, state: Any, questions: dict[str, Question]) -> BackendResponse:
        return run_coro_sync(self.run(state, questions))

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name}>"


def run_coro_sync(coro):
    """Run a coroutine from sync code, even if an event loop is already running (notebooks)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


def empty_usage() -> Usage:
    return Usage()
