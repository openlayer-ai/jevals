"""OpenAI Agents SDK. `pip install "jevals[openai-agents]"`.

from jevals.integrations.openai_agents import input_guardrail, output_guardrail, guard_tools

agent = Agent(
    name="support",
    instructions=SYSTEM_PROMPT,
    tools=guard_tools([lookup_order, issue_refund], before=tool_gate, after=ingress_gate, on_escalate=ask_human),
    input_guardrails=[input_guardrail(Gate(PromptInjection(block_below=0.5), PHI(action="redact")))],
    output_guardrails=[output_guardrail(Gate(SystemPromptLeakage(block_below=0.5), PII()))],
)
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Sequence
from typing import Any

from .._gate import Blocked, Decision, Gate


def _messages_from_context(ctx: Any, agent: Any = None) -> list[dict[str, Any]]:
    """Best-effort: pull the conversation out of a RunContextWrapper / ToolContext."""
    msgs: list[dict[str, Any]] = []
    instr = getattr(agent, "instructions", None)
    if isinstance(instr, str):
        msgs.append({"role": "system", "content": instr})
    for attr in ("messages", "input", "history", "items"):
        v = getattr(ctx, attr, None) or getattr(getattr(ctx, "context", None), attr, None)
        if v:
            if isinstance(v, str):
                msgs.append({"role": "user", "content": v})
            elif isinstance(v, list):
                msgs.extend(m for m in v if isinstance(m, dict))
            break
    return msgs


def input_guardrail(gate: Gate, name: str | None = None):
    """Wrap a Gate as an Agents SDK input guardrail. Trips on block. Modify rewrites nothing
    (the SDK has no hook for that on input), but the redacted value is in output_info."""
    from agents import GuardrailFunctionOutput  # type: ignore
    from agents import input_guardrail as _ig

    @_ig(name=name or f"jevals:{gate.name}")
    async def guardrail(ctx: Any, agent: Any, user_input: Any) -> Any:
        text = user_input if isinstance(user_input, str) else json.dumps(user_input, default=str)
        msgs = _messages_from_context(ctx, agent)
        msgs.append({"role": "user", "content": text})
        d = await gate.acheck({"messages": msgs, "input": text})
        return GuardrailFunctionOutput(
            output_info=d.model_dump(exclude={"results"}), tripwire_triggered=d.action == "block"
        )

    return guardrail


def output_guardrail(gate: Gate, name: str | None = None):
    from agents import GuardrailFunctionOutput  # type: ignore
    from agents import output_guardrail as _og

    @_og(name=name or f"jevals:{gate.name}")
    async def guardrail(ctx: Any, agent: Any, output: Any) -> Any:
        text = (
            output
            if isinstance(output, str)
            else getattr(output, "response", None) or json.dumps(output, default=str)
        )
        msgs = _messages_from_context(ctx, agent)
        msgs.append({"role": "assistant", "content": text})
        d = await gate.acheck({"messages": msgs, "output": text})
        return GuardrailFunctionOutput(
            output_info=d.model_dump(exclude={"results"}), tripwire_triggered=d.action == "block"
        )

    return guardrail


def guard_tools(
    tools: Sequence[Any],
    before: Gate | None = None,
    after: Gate | None = None,
    on_escalate: Callable[[Decision, str, dict[str, Any]], Any] | None = None,
    on_block: str = "message",
) -> list[Any]:
    """Wrap FunctionTools so `before` runs on the proposed call and `after` on the result.

    on_escalate(decision, tool_name, args) is awaited if async; its return value is used as
    the tool output (e.g. "Escalated to a human, ticket #123"). If not given, escalations
    proceed. on_block: "message" returns a short message to the model, "raise" raises Blocked.
    """
    out = []
    for t in tools:
        out.append(_guard_one(t, before, after, on_escalate, on_block))
    return out


def _guard_one(tool: Any, before: Gate | None, after: Gate | None, on_escalate: Any, on_block: str) -> Any:
    from agents import FunctionTool  # type: ignore

    if not isinstance(tool, FunctionTool):
        return tool
    inner = tool.on_invoke_tool
    name = tool.name

    async def on_invoke(ctx: Any, args_json: str) -> Any:
        try:
            args = json.loads(args_json) if args_json else {}
        except json.JSONDecodeError:
            args = {"raw": args_json}
        msgs = _messages_from_context(ctx)
        if before is not None:
            d = await before.acheck({"messages": msgs, "tool_call": {"name": name, "args": args}})
            if d.action == "block":
                if on_block == "raise":
                    raise Blocked(d)
                return f"Tool call blocked by policy: {d.reason}"
            if d.action == "escalate" and on_escalate is not None:
                r = on_escalate(d, name, args)
                if inspect.isawaitable(r):
                    r = await r
                return r if r is not None else f"Escalated for human review: {d.reason}"
        result = await inner(ctx, args_json)
        if after is not None:
            d = await after.acheck(
                {
                    "messages": msgs,
                    "tool_result": result if isinstance(result, str) else json.dumps(result, default=str),
                }
            )
            if d.action == "block":
                if on_block == "raise":
                    raise Blocked(d)
                return f"Tool result withheld by policy: {d.reason}"
            if d.action == "modify" and d.value is not None:
                return d.value
        return result

    return FunctionTool(
        name=tool.name,
        description=tool.description,
        params_json_schema=tool.params_json_schema,
        on_invoke_tool=on_invoke,
        strict_json_schema=getattr(tool, "strict_json_schema", True),
    )


def trace_from_result(result: Any, agent: Any = None) -> dict[str, Any]:
    """Turn a RunResult into a jevals sample: {"messages", "tools"}."""
    items = (
        result.to_input_list() if hasattr(result, "to_input_list") else list(getattr(result, "new_items", []))
    )
    messages: list[dict[str, Any]] = []
    instr = getattr(agent, "instructions", None) or getattr(
        getattr(result, "last_agent", None), "instructions", None
    )
    if isinstance(instr, str):
        messages.append({"role": "system", "content": instr})
    pending_calls: list[dict[str, Any]] = []
    for it in items:
        d = it if isinstance(it, dict) else getattr(it, "raw_item", None) or {}
        if hasattr(d, "model_dump"):
            d = d.model_dump()
        t = d.get("type")
        if t == "function_call":
            pending_calls.append(
                {
                    "id": d.get("call_id"),
                    "function": {"name": d.get("name"), "arguments": d.get("arguments", "{}")},
                }
            )
        elif t == "function_call_output":
            if pending_calls:
                messages.append({"role": "assistant", "content": None, "tool_calls": pending_calls})
                pending_calls = []
            messages.append({"role": "tool", "tool_call_id": d.get("call_id"), "content": d.get("output")})
        elif d.get("role"):
            if pending_calls:
                messages.append({"role": "assistant", "content": None, "tool_calls": pending_calls})
                pending_calls = []
            content = d.get("content")
            messages.append({"role": d["role"], "content": content})
    if pending_calls:
        messages.append({"role": "assistant", "content": None, "tool_calls": pending_calls})
    tools = []
    for t in getattr(agent, "tools", []) or getattr(getattr(result, "last_agent", None), "tools", []) or []:
        tools.append(
            {
                "name": getattr(t, "name", ""),
                "description": getattr(t, "description", ""),
                "parameters": getattr(t, "params_json_schema", {}),
            }
        )
    return {"messages": messages, "tools": tools}
