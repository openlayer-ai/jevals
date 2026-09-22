"""The sample: a plain dict with attribute access and fields derived from `messages`.

You pass `{"messages": [...], "tools": [...]}` (or `input`/`output`/`contexts`
for single-turn work). Evals read `s.final_answer`, `s.tool_calls`,
`s.tool_results`, `s.user_messages`, and so on. Those are computed here from
whatever message format you happened to have: OpenAI chat, Anthropic content
blocks, or LangChain message objects.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from functools import cached_property
from typing import Any


class ToolCall(dict):
    """{"id", "name", "args", "result"}; `result` is filled when a matching tool message exists."""

    @property
    def id(self) -> str | None:
        return self.get("id")

    @property
    def name(self) -> str:
        return self.get("name", "")

    @property
    def args(self) -> dict[str, Any]:
        return self.get("args") or {}

    @property
    def result(self) -> Any:
        return self.get("result")


def _parse_args(a: Any) -> dict[str, Any]:
    if a is None:
        return {}
    if isinstance(a, str):
        try:
            v = json.loads(a)
            return v if isinstance(v, dict) else {"value": v}
        except json.JSONDecodeError:
            return {"raw": a}
    if isinstance(a, Mapping):
        return dict(a)
    return {"value": a}


def _text(content: Any) -> str:
    """Flatten any content shape into text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, Mapping):
                t = part.get("type")
                if t in ("text", "output_text", "input_text"):
                    out.append(str(part.get("text", "")))
                elif t == "tool_result":
                    out.append(_text(part.get("content")))
            else:
                txt = getattr(part, "text", None)
                if txt:
                    out.append(str(txt))
        return "\n".join(x for x in out if x)
    if isinstance(content, Mapping):
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def normalize_messages(messages: Iterable[Any]) -> list[dict[str, Any]]:
    """Normalize to a list of {role, content, tool_calls?, tool_call_id?, name?}.

    Accepts OpenAI chat dicts, Anthropic messages with content blocks, and
    LangChain BaseMessage objects (duck-typed on `.type` / `.content`).
    Anthropic `tool_use` and `tool_result` blocks are split into assistant
    tool_calls and separate `tool` role messages so downstream code sees one shape.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        if m is None:
            continue

        # LangChain BaseMessage (or anything shaped like one)
        if not isinstance(m, Mapping) and hasattr(m, "content") and hasattr(m, "type"):
            lc_type = getattr(m, "type", "")
            role = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}.get(
                lc_type, lc_type
            )
            msg: dict[str, Any] = {"role": role, "content": _text(m.content)}
            tcs = getattr(m, "tool_calls", None) or []
            if tcs:
                msg["tool_calls"] = [
                    {"id": tc.get("id"), "name": tc.get("name"), "args": _parse_args(tc.get("args"))}
                    for tc in tcs
                    if isinstance(tc, Mapping)
                ]
            if role == "tool":
                msg["tool_call_id"] = getattr(m, "tool_call_id", None)
                msg["name"] = getattr(m, "name", None)
            out.append(msg)
            continue

        if not isinstance(m, Mapping):
            out.append({"role": "user", "content": _text(m)})
            continue

        role = m.get("role", "user")
        content = m.get("content")

        # Anthropic content blocks: split tool_use / tool_result
        if isinstance(content, list) and any(
            isinstance(p, Mapping) and p.get("type") in ("tool_use", "tool_result") for p in content
        ):
            text_parts = [
                p
                for p in content
                if not (isinstance(p, Mapping) and p.get("type") in ("tool_use", "tool_result"))
            ]
            uses = [p for p in content if isinstance(p, Mapping) and p.get("type") == "tool_use"]
            results = [p for p in content if isinstance(p, Mapping) and p.get("type") == "tool_result"]
            if text_parts or uses:
                msg = {"role": role, "content": _text(text_parts)}
                if uses:
                    msg["tool_calls"] = [
                        {"id": u.get("id"), "name": u.get("name"), "args": _parse_args(u.get("input"))}
                        for u in uses
                    ]
                out.append(msg)
            for r in results:
                out.append(
                    {"role": "tool", "tool_call_id": r.get("tool_use_id"), "content": _text(r.get("content"))}
                )
            continue

        msg = {"role": role, "content": _text(content)}
        # OpenAI tool_calls
        tcs = m.get("tool_calls")
        if tcs:
            norm = []
            for tc in tcs:
                if not isinstance(tc, Mapping):
                    continue
                fn = tc.get("function") or {}
                norm.append(
                    {
                        "id": tc.get("id"),
                        "name": tc.get("name") or fn.get("name"),
                        "args": _parse_args(tc.get("args", tc.get("arguments", fn.get("arguments")))),
                    }
                )
            msg["tool_calls"] = norm
        # legacy OpenAI function_call
        if m.get("function_call"):
            fc = m["function_call"]
            msg["tool_calls"] = [
                {"id": None, "name": fc.get("name"), "args": _parse_args(fc.get("arguments"))}
            ]
        if role in ("tool", "function"):
            msg["role"] = "tool"
            msg["tool_call_id"] = m.get("tool_call_id")
            msg["name"] = m.get("name")
        out.append(msg)
    return out


def normalize_tools(tools: Any) -> list[dict[str, Any]]:
    """Normalize tool schemas to [{name, description, parameters}]."""
    if not tools:
        return []
    out = []
    for t in tools:
        if isinstance(t, str):
            out.append({"name": t, "description": "", "parameters": {}})
        elif isinstance(t, Mapping):
            fn = (
                t.get("function")
                if t.get("type") == "function" and isinstance(t.get("function"), Mapping)
                else t
            )
            out.append(
                {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", "") or "",
                    "parameters": fn.get("parameters")
                    or fn.get("input_schema")
                    or fn.get("params_json_schema")
                    or {},
                }
            )
        else:  # objects: LangChain tools, OpenAI Agents FunctionTool, plain callables
            out.append(
                {
                    "name": getattr(t, "name", None) or getattr(t, "__name__", str(t)),
                    "description": getattr(t, "description", None) or (getattr(t, "__doc__", "") or ""),
                    "parameters": getattr(t, "params_json_schema", None)
                    or getattr(t, "args_schema", None)
                    or {},
                }
            )
    return out


class Sample(dict):
    """A dict with attribute access and derived fields.

    Raw keys win: if you pass `final_answer` explicitly, that's what evals get.
    """

    def __init__(self, data: Mapping[str, Any] | None = None, /, **kw: Any):
        super().__init__()
        if data:
            self.update(data)
        self.update(kw)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        if name in self:
            return self[name]
        # cached_property access
        cls_attr = getattr(type(self), name, None)
        if isinstance(cls_attr, cached_property):
            return cls_attr.__get__(self, type(self))
        return None

    def has(self, *keys: str) -> bool:
        return all(getattr(self, k, None) not in (None, "", [], {}) for k in keys)

    # ------------------------------------------------------------------ derived

    @cached_property
    def messages(self) -> list[dict[str, Any]]:
        raw = self.get("messages")
        if raw is None:
            msgs = []
            if self.get("system_prompt"):
                msgs.append({"role": "system", "content": str(self["system_prompt"])})
            if self.get("input") is not None:
                msgs.append({"role": "user", "content": _text(self["input"])})
            if self.get("output") is not None:
                msgs.append({"role": "assistant", "content": _text(self["output"])})
            return msgs
        return normalize_messages(raw)

    @cached_property
    def tools(self) -> list[dict[str, Any]]:
        return normalize_tools(self.get("tools"))

    @cached_property
    def tool_names(self) -> list[str]:
        return [t["name"] for t in self.tools]

    @cached_property
    def system_prompt(self) -> str:
        if self.get("system_prompt") is not None:
            return str(self["system_prompt"])
        return "\n".join(m["content"] for m in self.messages if m["role"] == "system")

    @cached_property
    def user_messages(self) -> list[str]:
        return [m["content"] for m in self.messages if m["role"] == "user" and m.get("content")]

    @cached_property
    def input(self) -> str:
        if self.get("input") is not None:
            return _text(self["input"])
        return self.user_messages[0] if self.user_messages else ""

    @cached_property
    def last_user_message(self) -> str:
        return self.user_messages[-1] if self.user_messages else self.input

    @cached_property
    def final_answer(self) -> str:
        if self.get("final_answer") is not None:
            return _text(self["final_answer"])
        if self.get("output") is not None:
            return _text(self["output"])
        for m in reversed(self.messages):
            if m["role"] == "assistant" and m.get("content"):
                return m["content"]
        return ""

    @cached_property
    def output(self) -> str:
        return self.final_answer

    @cached_property
    def tool_calls(self) -> list[ToolCall]:
        if self.get("tool_calls") is not None:
            return [
                ToolCall(
                    id=tc.get("id"),
                    name=tc.get("name") or (tc.get("function") or {}).get("name"),
                    args=_parse_args(
                        tc.get("args", tc.get("arguments", (tc.get("function") or {}).get("arguments")))
                    ),
                    result=tc.get("result", tc.get("output")),
                )
                for tc in self["tool_calls"]
                if isinstance(tc, Mapping)
            ]
        calls: list[ToolCall] = []
        by_id: dict[str, ToolCall] = {}
        pending: list[ToolCall] = []
        for m in self.messages:
            if m["role"] == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    call = ToolCall(
                        id=tc.get("id"), name=tc.get("name"), args=tc.get("args") or {}, result=None
                    )
                    calls.append(call)
                    pending.append(call)
                    if tc.get("id"):
                        by_id[tc["id"]] = call
            elif m["role"] == "tool":
                target = by_id.get(m.get("tool_call_id") or "")
                if target is None and pending:
                    target = pending[0]
                if target is not None:
                    target["result"] = m.get("content")
                    if target in pending:
                        pending.remove(target)
        return calls

    @cached_property
    def tool_results(self) -> list[Any]:
        if self.get("tool_results") is not None:
            return list(self["tool_results"])
        return [tc.result for tc in self.tool_calls if tc.result is not None]

    @cached_property
    def tool_result(self) -> Any:
        if self.get("tool_result") is not None:
            return self["tool_result"]
        return self.tool_results[-1] if self.tool_results else None

    @cached_property
    def tool_call(self) -> ToolCall | None:
        raw = self.get("tool_call")
        if raw is not None:
            if isinstance(raw, Mapping):
                fn = raw.get("function") or {}
                return ToolCall(
                    id=raw.get("id"),
                    name=raw.get("name") or fn.get("name"),
                    args=_parse_args(raw.get("args", raw.get("arguments", fn.get("arguments")))),
                    result=raw.get("result"),
                )
            return ToolCall(
                id=getattr(raw, "id", None),
                name=getattr(raw, "name", None) or getattr(getattr(raw, "function", None), "name", None),
                args=_parse_args(
                    getattr(raw, "args", None)
                    or getattr(raw, "arguments", None)
                    or getattr(raw, "input", None)
                ),
                result=None,
            )
        return self.tool_calls[-1] if self.tool_calls else None

    @cached_property
    def steps(self) -> list[dict[str, Any]]:
        """Assistant turns and tool calls in order, compact, for `recent steps` style state."""
        out = []
        for m in self.messages:
            if m["role"] == "assistant":
                if m.get("tool_calls"):
                    for tc in m["tool_calls"]:
                        out.append({"type": "tool_call", "name": tc.get("name"), "args": tc.get("args")})
                if m.get("content"):
                    out.append({"type": "assistant", "content": m["content"]})
            elif m["role"] == "tool":
                out.append({"type": "tool_result", "content": m.get("content")})
        return out

    @cached_property
    def contexts(self) -> list[str]:
        raw = self.get("contexts") or self.get("retrieved_contexts") or self.get("context")
        if raw is None:
            return []
        if isinstance(raw, str):
            return [raw]
        return [_text(c) for c in raw]

    @cached_property
    def expected(self) -> str | None:
        v = self.get("expected", self.get("reference", self.get("expected_output")))
        return _text(v) if v is not None else None

    @cached_property
    def goal(self) -> str:
        return _text(self.get("goal")) if self.get("goal") else self.input

    def to_dict(self) -> dict[str, Any]:
        return dict(self)


def as_sample(x: Any) -> Sample:
    if isinstance(x, Sample):
        return x
    if isinstance(x, Mapping):
        return Sample(x)
    if isinstance(x, str):
        return Sample(output=x)
    if isinstance(x, (list, tuple)):
        return Sample(messages=list(x))
    raise TypeError(f"cannot build a sample from {type(x).__name__}")
