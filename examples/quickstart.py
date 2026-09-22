"""The README example, runnable. `python examples/quickstart.py` (uses backend=mock unless a key is set)."""

import json
import os

from jevals import evaluate
from jevals.agent import Grounded, Quality, StayedInScope, ToolChoice, UsedToolResult
from jevals.quality import AnswerRelevancy, Completeness
from jevals.security import PHI, IndirectInjection

tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Web search for live information",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        },
    }
]

messages = [
    {"role": "system", "content": "You are a weather assistant."},
    {"role": "user", "content": "What's the weather in San Francisco today?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "search", "arguments": json.dumps({"q": "San Francisco weather today"})},
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "c1",
        "content": "San Francisco, today: sunny, high 68F, low 54F, wind 12 mph from the west.",
    },
    {
        "role": "assistant",
        "content": "It's sunny in San Francisco today with a high of 68F and a low of 54F. Winds are light from the west, and it will stay sunny all week.",
    },
]

backend = (
    None
    if any(
        os.environ.get(k)
        for k in (
            "TYPESAFE_API_KEY",
            "AI_GATEWAY_API_KEY",
            "KEV_BASE_URL",
            "OPENROUTER_API_KEY",
            "JEVALS_BACKEND",
        )
    )
    else "mock"
)

r = evaluate(
    {"messages": messages, "tools": tools},
    [
        ToolChoice(
            options={
                "searched_appropriately": "Called search because the question needed live data",
                "searched_unnecessarily": "Called search for something it already knew or the user didn't ask",
                "failed_to_search": "Answered from memory when the question needed live data",
            }
        ),
        UsedToolResult(),
        Grounded(),
        StayedInScope(),
        Quality(levels=["unhelpful", "partially", "adequate", "good", "excellent"]),
        AnswerRelevancy(),
        Completeness(),
        IndirectInjection(),
        PHI(),
    ],
    backend=backend,
)

print(r.table())
if backend == "mock":
    print("\n(mock backend: set AI_GATEWAY_API_KEY or another key to see real probabilities)")
