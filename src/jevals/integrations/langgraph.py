"""LangGraph. `pip install "jevals[langgraph]"`.

    from jevals.integrations.langgraph import gate_node, trace_from_state

    graph.add_node("gate", gate_node(tool_gate, on_block="message"))
    graph.add_edge("agent", "gate")
    graph.add_conditional_edges("gate", route_after_gate, {"tools": "tools", "agent": "agent", "human": "human"})

`gate_node` reads the last AIMessage's tool_calls, runs the gate on each, and writes
`jevals_decisions` into state. With on_block="message" it also replaces blocked tool calls
with ToolMessages explaining the block, so the graph can continue without running them.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from .._gate import Decision, Gate
from .._sample import normalize_messages


def trace_from_state(
    state: dict[str, Any], tools: Any = None, messages_key: str = "messages"
) -> dict[str, Any]:
    msgs = normalize_messages(state.get(messages_key, []))
    return {"messages": msgs, "tools": tools or []}


def gate_node(
    gate: Gate,
    on_block: str = "message",
    on_escalate: Callable[[Decision, dict[str, Any]], Any] | None = None,
    messages_key: str = "messages",
    tools: Any = None,
):
    """A node that gates the tool calls proposed by the last AI message."""

    async def node(state: dict[str, Any]) -> dict[str, Any]:
        from langchain_core.messages import AIMessage, ToolMessage  # type: ignore

        msgs = state.get(messages_key, [])
        if not msgs or not isinstance(msgs[-1], AIMessage) or not msgs[-1].tool_calls:
            return {"jevals_decisions": []}
        last: Any = msgs[-1]
        conv = normalize_messages(msgs)
        decisions: list[dict[str, Any]] = []
        new_msgs: list[Any] = []
        kept_calls = []
        for tc in last.tool_calls:
            d = await gate.acheck(
                {
                    "messages": conv,
                    "tools": tools or [],
                    "tool_call": {"id": tc.get("id"), "name": tc.get("name"), "args": tc.get("args")},
                }
            )
            decisions.append(
                {"tool_call_id": tc.get("id"), "tool": tc.get("name"), **d.model_dump(exclude={"results"})}
            )
            if d.action == "block":
                if on_block == "message":
                    new_msgs.append(
                        ToolMessage(
                            content=f"Blocked by policy: {d.reason}",
                            tool_call_id=tc.get("id"),
                            name=tc.get("name"),
                        )
                    )
                    continue
            elif d.action == "escalate" and on_escalate is not None:
                r = on_escalate(d, tc)
                if hasattr(r, "__await__"):
                    r = await r
                new_msgs.append(
                    ToolMessage(
                        content=str(r or f"Escalated for review: {d.reason}"),
                        tool_call_id=tc.get("id"),
                        name=tc.get("name"),
                    )
                )
                continue
            kept_calls.append(tc)
        out: dict[str, Any] = {"jevals_decisions": decisions}
        if new_msgs:
            if kept_calls:
                # split: rewrite the AI message to only the kept calls, then append tool messages for the others
                last = AIMessage(content=last.content, tool_calls=kept_calls, id=last.id)
            out[messages_key] = ([last] if kept_calls else []) + new_msgs
            out["jevals_kept_tool_calls"] = kept_calls
        return out

    return node


def route_after_gate(state: dict[str, Any]) -> str:
    """Default router: 'tools' if any call survived, 'human' if any escalated without a handler, else 'agent'."""
    ds = state.get("jevals_decisions", [])
    if not ds:
        return "tools"
    if any(d["action"] == "escalate" for d in ds) and "jevals_kept_tool_calls" not in state:
        return "human"
    kept = state.get("jevals_kept_tool_calls")
    if kept is None:
        return "tools" if all(d["action"] in ("allow", "modify") for d in ds) else "agent"
    return "tools" if kept else "agent"


def guard_tool_result(gate: Gate):
    """Wrap a LangChain tool so its output passes through an ingress gate (redaction, injection)."""

    def wrap(tool: Any) -> Any:
        from langchain_core.tools import StructuredTool  # type: ignore

        async def run(**kwargs: Any) -> Any:
            out = await tool.ainvoke(kwargs) if hasattr(tool, "ainvoke") else tool.invoke(kwargs)
            text = out if isinstance(out, str) else json.dumps(out, default=str)
            d = await gate.acheck({"tool_result": text, "messages": []})
            if d.action == "block":
                return f"Tool result withheld by policy: {d.reason}"
            return d.value if d.action == "modify" else out

        return StructuredTool.from_function(
            coroutine=run,
            name=tool.name,
            description=tool.description,
            args_schema=getattr(tool, "args_schema", None),
        )

    return wrap
