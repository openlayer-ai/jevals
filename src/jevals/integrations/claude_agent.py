"""Claude Agent SDK / Claude Code hooks. `pip install "jevals[claude]"`.

In-process (Claude Agent SDK):

    from jevals.integrations.claude_agent import pre_tool_use, post_tool_use
    options = ClaudeAgentOptions(hooks={
        "PreToolUse":  [HookMatcher(matcher="Bash|Write|Edit", hooks=[pre_tool_use(tool_gate)])],
        "PostToolUse": [HookMatcher(hooks=[post_tool_use(ingress_gate)])],
    })

As a Claude Code hook command (settings.json):

    {"hooks": {"PreToolUse": [{"matcher": "Bash|Write", "hooks": [{"type": "command", "command": "jevals hook pre --evals agent.tool_call_risk"}]}]}}

The hook reads the JSON event on stdin and writes the permission decision on stdout.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from .._gate import Gate

_DECISION = {"allow": "allow", "modify": "allow", "escalate": "ask", "block": "deny"}


def _sample_from_event(event: dict[str, Any]) -> dict[str, Any]:
    transcript = event.get("transcript") or []
    messages = list(transcript) if isinstance(transcript, list) else []
    if not messages and event.get("prompt"):
        messages = [{"role": "user", "content": event["prompt"]}]
    s: dict[str, Any] = {"messages": messages}
    if "tool_name" in event:
        s["tool_call"] = {
            "id": event.get("tool_use_id"),
            "name": event["tool_name"],
            "args": event.get("tool_input") or {},
        }
    if "tool_response" in event:
        tr = event["tool_response"]
        s["tool_result"] = tr if isinstance(tr, str) else json.dumps(tr, default=str)
    return s


def pre_tool_use(gate: Gate):
    """PreToolUse hook: allow / ask / deny from the gate decision."""

    async def hook(input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> dict[str, Any]:
        d = await gate.acheck(_sample_from_event({**input_data, "tool_use_id": tool_use_id}))
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": _DECISION[d.action],
                "permissionDecisionReason": d.reason or f"jevals {gate.name}: {d.action}",
            }
        }

    return hook


def post_tool_use(gate: Gate):
    """PostToolUse hook: block feeds the reason back to the model; modify is reported as additional context."""

    async def hook(input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> dict[str, Any]:
        d = await gate.acheck(_sample_from_event({**input_data, "tool_use_id": tool_use_id}))
        if d.action == "block":
            return {"decision": "block", "reason": d.reason}
        if d.action == "modify":
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": f"Tool result contained sensitive data; redacted version: {d.value}",
                }
            }
        if d.action == "escalate":
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": f"jevals flagged this result: {d.reason}",
                }
            }
        return {}

    return hook


def run_hook_command(gate: Gate, phase: str = "pre") -> int:
    """stdin JSON -> stdout JSON, for `jevals hook`. Returns the process exit code."""
    event = json.loads(sys.stdin.read() or "{}")
    d = gate.check(_sample_from_event(event))
    if phase == "pre":
        out = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": _DECISION[d.action],
                "permissionDecisionReason": d.reason or f"jevals: {d.action}",
            }
        }
        print(json.dumps(out))
        return 0
    if d.action == "block":
        print(json.dumps({"decision": "block", "reason": d.reason}))
        return 0
    if d.action in ("modify", "escalate"):
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PostToolUse",
                        "additionalContext": d.reason or "jevals: redacted",
                    }
                }
            )
        )
    return 0
