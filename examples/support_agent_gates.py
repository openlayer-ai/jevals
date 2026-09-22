"""A support agent loop with three gates: tool-call risk before each tool, ingress scan on
every tool result, and a loop check every step. No framework; this is the "your own loop" case.

Run with backend=mock: `python examples/support_agent_gates.py`
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from jevals import Blocked, Gate, gate, load_eval
from jevals.agent import LoopDetection
from jevals.security import PHI, GoalHijacking, IndirectInjection

HERE = Path(__file__).parent
BACKEND = (
    None
    if any(
        os.environ.get(k)
        for k in ("TYPESAFE_API_KEY", "AI_GATEWAY_API_KEY", "KEV_BASE_URL", "OPENROUTER_API_KEY")
    )
    else "mock"
)

tool_gate = Gate(load_eval(HERE / "evals" / "tool_call_risk.yaml"), backend=BACKEND)
ingress_gate = Gate(
    IndirectInjection(surface="tool_result", block_below=0.5),
    GoalHijacking(block_below=0.5),
    PHI(surface="tool_result", action="redact"),
    backend=BACKEND,
)
loop_gate = Gate(LoopDetection(window=6, escalate_below=0.4), backend=BACKEND)


# ---- fake tools ---------------------------------------------------------------------------

ORDERS = {"A123": {"id": "A123", "status": "shipped", "eta": "Tuesday", "customer": "jane@example.com"}}


async def lookup_order(order_id: str) -> str:
    return json.dumps(ORDERS.get(order_id, {"error": "not found"}))


async def issue_refund(order_id: str, amount: float) -> str:
    return f"refunded {amount} on {order_id}"


async def send_email(to: str, body: str) -> str:
    return f"sent to {to}"


TOOLS = {"lookup_order": lookup_order, "issue_refund": issue_refund, "send_email": send_email}


async def ask_human(decision) -> str:
    print(f"  -> paging a human: {decision.reason}")
    return "Escalated to a human agent; they will follow up."


# ---- the gated dispatch -------------------------------------------------------------------


@gate(tool_gate, on_escalate=ask_human)
async def call_tool(call: dict, messages: list[dict]) -> str:
    out = await TOOLS[call["name"]](**call["args"])
    d = await ingress_gate.acheck({"tool_result": out, "messages": messages})
    if d.action == "block":
        raise Blocked(d)
    return d.value  # redacted if PHI was found, otherwise the original


async def main() -> None:
    messages = [
        {"role": "system", "content": "You are a support agent for Acme."},
        {"role": "user", "content": "Where is order A123?"},
    ]
    proposed = [
        {"name": "lookup_order", "args": {"order_id": "A123"}},
        {"name": "issue_refund", "args": {"order_id": "A123", "amount": 500.0}},
    ]
    for call in proposed:
        print(f"{call['name']}({call['args']})")
        try:
            out = await call_tool(call=call, messages=messages)
            print(f"  -> {out}")
        except Blocked as e:
            print(f"  -> blocked: {e.decision.reason}")
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call["name"],
                        "function": {"name": call["name"], "arguments": json.dumps(call["args"])},
                    }
                ],
            }
        )
        messages.append({"role": "tool", "tool_call_id": call["name"], "content": "..."})
        d = await loop_gate.acheck({"messages": messages})
        if d.action != "allow":
            print(f"  loop gate: {d}")


if __name__ == "__main__":
    asyncio.run(main())
