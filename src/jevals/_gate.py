"""Gates: run evals at runtime and decide allow / block / escalate / modify."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Sequence
from typing import Any

from pydantic import BaseModel, Field

from ._eval import Action, Eval, Result
from ._runner import EvalReport, aevaluate
from ._sample import Sample, as_sample
from ._types import Usage
from .backends import Backend
from .backends._base import run_coro_sync

_ORDER: dict[Action, int] = {"block": 3, "escalate": 2, "modify": 1, "allow": 0}


class Decision(BaseModel):
    action: Action
    reasons: list[str] = Field(default_factory=list)
    results: list[Result] = Field(default_factory=list)
    value: Any = (
        None  # the payload to use: redacted text when modified, the original otherwise, None when blocked
    )
    usage: Usage = Field(default_factory=Usage)

    @property
    def allowed(self) -> bool:
        return self.action in ("allow", "modify")

    @property
    def reason(self) -> str:
        return "; ".join(self.reasons)

    def __bool__(self) -> bool:
        return self.allowed

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        for r in self.results:
            if r.name == name:
                return r
        raise AttributeError(name)

    def __str__(self) -> str:
        return self.action + (f": {self.reason}" if self.reasons else "")


class Blocked(Exception):
    def __init__(self, decision: Decision):
        self.decision = decision
        super().__init__(str(decision))


def _payload(s: Sample) -> Any:
    for k in ("tool_result", "output", "final_answer", "input", "text"):
        if s.get(k) is not None:
            return s[k]
    return None


class Gate:
    """Evals plus a policy, run at request time.

        gate = Gate(ToolCallRisk(), PII(action="redact"))
        d = gate.check({"messages": messages, "tool_call": call})
        d.action      # "allow" | "block" | "escalate" | "modify"
        d.value       # redacted payload on modify, original otherwise
        d.reasons     # ["tool_call_risk: escalate (approve=0.41 ...)"]

    Per-eval `decide()` hooks run first (block_below / escalate_below / action="redact" ...),
    then `policy(report)` if given. `on_error` is what to do if the backend fails:
    "allow" (fail open, default), "block", or "escalate". `on_block="raise"` makes
    check() raise Blocked instead of returning.
    """

    def __init__(
        self,
        *evals: Eval | Sequence[Eval],
        policy: Callable[[EvalReport], Action | None] | None = None,
        backend: str | Backend | None = None,
        on_error: Action = "allow",
        on_block: str = "return",
        name: str = "gate",
    ):
        flat: list[Eval] = []
        for e in evals:
            if isinstance(e, Eval):
                flat.append(e)
            else:
                flat.extend(e)
        self.evals = flat
        self.policy = policy
        self.backend = backend
        self.on_error = on_error
        self.on_block = on_block
        self.name = name

    async def acheck(self, sample: Any) -> Decision:
        s = as_sample(sample)
        report = await aevaluate(s, self.evals, self.backend)
        d = self._decide(report, s)
        if d.action == "block" and self.on_block == "raise":
            raise Blocked(d)
        return d

    def check(self, sample: Any) -> Decision:
        return run_coro_sync(self.acheck(sample))

    __call__ = check

    def _decide(self, report: EvalReport, s: Sample) -> Decision:
        worst: Action = "allow"
        reasons: list[str] = []
        value: Any = _payload(s)
        by_name = {ev.name: ev for ev in self.evals}
        for r in report.results:
            ev = by_name.get(r.name)
            if r.error:
                if _ORDER[self.on_error] > _ORDER[worst]:
                    worst = self.on_error
                reasons.append(f"{r.name}: error ({r.error[:80]})")
                continue
            if r.skipped or ev is None:
                continue
            a = r.action or ev.decide(r, s)
            if a is None or a == "allow":
                continue
            r.action = a
            if _ORDER[a] > _ORDER[worst]:
                worst = a
            if a == "modify" and r.value is not None:
                value = r.value
            reasons.append(f"{r.name}: {a}" + (f" ({r.detail})" if r.detail else ""))
        if self.policy is not None:
            p = self.policy(report)
            if p is not None and _ORDER[p] > _ORDER[worst]:
                worst = p
                reasons.append(f"policy: {p}")
        if worst == "block":
            value = None
        return Decision(
            action=worst, reasons=reasons, results=report.results, value=value, usage=report.usage
        )


def gate(
    *evals: Eval | Gate | Sequence[Eval],
    sample: Callable[..., Any] | None = None,
    on_block: Callable[[Decision], Any] | str = "raise",
    on_escalate: Callable[[Decision], Any] | None = None,
    backend: str | Backend | None = None,
    on_error: Action = "allow",
):
    """Decorate a tool or a tool-dispatch function. Its inputs are evaluated before it runs.

        @gate(ToolCallRisk(), on_escalate=ask_human)
        async def call_tool(call, messages): ...

        @gate(tool_gate)                      # an existing Gate works too
        def refund(order_id: str, amount: float, messages=None): ...

    The sample is built from the call's arguments: a parameter named `call` or
    `tool_call` becomes `tool_call`; `messages`, `tools`, `tool_result`, `input`,
    `output` pass through; anything else becomes `tool_call.args` under the function's
    name. Pass `sample=lambda **kw: {...}` to build it yourself.
    on_block: "raise" (Blocked), "none" (return None), or a callable receiving the Decision.
    on_escalate: callable receiving the Decision; its return value is returned instead of running.
    """
    if len(evals) == 1 and isinstance(evals[0], Gate):
        g = evals[0]
    else:
        g = Gate(*[e for e in evals if not isinstance(e, Gate)], backend=backend, on_error=on_error)  # type: ignore[arg-type]

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(fn)
        passthrough = (
            "messages",
            "tools",
            "tool_result",
            "input",
            "output",
            "contexts",
            "system_prompt",
            "goal",
        )

        def build(args: tuple, kwargs: dict) -> dict[str, Any]:
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            kw = dict(bound.arguments)
            if sample is not None:
                return sample(**kw)
            out: dict[str, Any] = {}
            call = kw.pop("call", None) or kw.pop("tool_call", None)
            for k in passthrough:
                if k in kw:
                    out[k] = kw.pop(k)
            if call is not None:
                out["tool_call"] = call
            else:
                out["tool_call"] = {
                    "name": fn.__name__,
                    "args": {k: v for k, v in kw.items() if _jsonable(v)},
                }
            out.setdefault("messages", [])
            return out

        def handle(d: Decision) -> tuple[Any, bool]:
            if d.action == "block":
                if on_block == "raise":
                    raise Blocked(d)
                if on_block in ("none", None):
                    return None, True
                return on_block(d), True  # type: ignore[operator]
            if d.action == "escalate" and on_escalate is not None:
                return on_escalate(d), True
            return None, False

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def aw(*args: Any, **kwargs: Any) -> Any:
                d = await g.acheck(build(args, kwargs))
                out, stop = handle(d)
                if stop:
                    return await out if inspect.isawaitable(out) else out
                return await fn(*args, **kwargs)

            aw.gate = g  # type: ignore[attr-defined]
            return aw

        @functools.wraps(fn)
        def w(*args: Any, **kwargs: Any) -> Any:
            d = g.check(build(args, kwargs))
            out, stop = handle(d)
            if stop:
                return out
            return fn(*args, **kwargs)

        w.gate = g  # type: ignore[attr-defined]
        return w

    return decorate


def _jsonable(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool, list, dict, tuple, type(None)))
