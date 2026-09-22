"""`jevals docs --llm`: a compact reference to paste into a coding agent's context."""

from __future__ import annotations

from .._registry import builtin_evals


def llm_docs() -> str:
    evals = builtin_evals()
    by_cat: dict[str, list[str]] = {}
    for name, cls in evals.items():
        desc = (cls.__doc__ or "").strip().split("\n")[0]
        by_cat.setdefault(cls.category, []).append(f"  {cls.category}.{name:<22} {desc}")
    listing = "\n".join(f"{cat}\n" + "\n".join(sorted(rows)) for cat, rows in by_cat.items())
    return f"""# jevals, in one page

Evals and guardrails for agents. Each eval is typed questions over a small state; all evals on a
sample are packed into ONE request to a System One model (Jev / Kev / Laya) or an emulating LLM.

## Setup
pip install jevals            # + "jevals[pii]" for Presidio, "jevals[mcp]" for the MCP server
export AI_GATEWAY_API_KEY=... # Jev via Vercel. Or TYPESAFE_API_KEY, KEV_BASE_URL, OPENROUTER_API_KEY.
                              # backend="mock" in tests. JEVALS_BACKEND overrides autodetect.

## Sample (a plain dict)
{{"messages": [...], "tools": [...]}}          # messages in OpenAI chat format (Anthropic / LangChain accepted)
optional: "tool_call" (the proposed call to gate), "tool_result", "contexts", "expected", "input", "output",
          "system_prompt", "goal", "expected_tool_calls", "allowed_topics"
Derived fields evals read: s.input, s.final_answer, s.tool_calls, s.tool_results, s.tool_call, s.user_messages,
s.system_prompt, s.steps, s.contexts, s.expected, s.goal

## Run
from jevals import evaluate, aevaluate, evaluate_dataset, Gate, gate, load_eval
r = evaluate(sample, [ToolChoice(), Grounded(), IndirectInjection(), PII()])   # sync; aevaluate for async
r.grounded.score / .passed / .probability / .answer / .detail / .evidence;  r.usage;  print(r.table())
rep = evaluate_dataset("traces.jsonl", evals); print(rep.table()); rep.write_jsonl("out.jsonl")

## Gate (runtime)
g = Gate(ToolCallRisk(), IndirectInjection(surface="tool_result", block_below=0.5), PII(action="redact"))
d = g.check({{"messages": msgs, "tool_call": {{"name": "refund", "args": {{...}}}}}})
d.action in ("allow", "block", "escalate", "modify"); d.reasons; d.value (redacted payload on modify)
@gate(g, on_escalate=ask_human)  async def call_tool(call, messages): ...
Per-eval knobs: block_below=, escalate_below= (on the 0..1 safe score); PII/PHI/Secrets: action="redact"|"block"
Integrations: jevals.integrations.openai_agents (input_guardrail, output_guardrail, guard_tools),
              jevals.integrations.langgraph (gate_node, route_after_gate), jevals.integrations.claude_agent (pre_tool_use, post_tool_use)

## Built-in evals (constructor kwargs: threshold=, block_below=, escalate_below=; many take surface="input"|"output"|"tool_result")
{listing}

## Write your own (YAML; `jevals validate file.yaml`; `jevals schema` for the JSON Schema)
name: refund_policy
requires: [messages]
state:                                   # what the model sees. $.path into the sample, or literals
  policy: "Refunds within 30 days on Pro plans."
  response: $.final_answer
  recent: $.messages[-4:]
questions:
  promises: {{type: noul, instructions: "Does response promise a refund?"}}
  eligible:
    type: noul
    instructions: "Per policy and the conversation, is the customer eligible?"
    criteria: {{"true": "Within 30 days on Pro", "false": "Out of window or wrong plan"}}
score: "1 - promises * (1 - eligible)"    # expressions over question ids; choice -> action.approve, action.choice
pass: "score >= 0.7"
policy: {{block_if: "promises > 0.8 and eligible < 0.3"}}   # or allow_if / escalate_if / else: escalate

Question types: noul (yes/no -> probability), choice (criteria: {{key: description}} -> probabilities per key),
score (criteria: [lowest..highest] -> expected level). Describe options, don't label them. One atomic question
per thing. Keep state small: you pay for input tokens only.

Python equivalent: subclass jevals.Eval with state(s) -> dict, questions(s) -> {{id: Noul|Choice|Score}},
reduce(answers, s) -> Result(score=, passed=, detail=, evidence=). Or NoulEval / ScoreEval / ChoiceEval for one question.

## CLI
jevals run traces.jsonl --evals agent.tool_choice,agent.grounded,security.phi,evals/mine.yaml [--out results.jsonl]
jevals list | jevals describe grounded | jevals validate evals/*.yaml | jevals schema | jevals docs --llm
jevals calibrate labeled.jsonl --eval evals/tool_call_risk.yaml --label human_decision
jevals mcp --install
"""
