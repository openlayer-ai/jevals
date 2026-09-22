import json

import pytest

from jevals import MockBackend, Sample, evaluate
from jevals.agent import (
    ArgumentValidity,
    GoalCompletion,
    LoopDetection,
    Quality,
    StepProgress,
    ToolCallF1,
    ToolCallRisk,
    TrajectoryMatch,
)
from jevals.quality import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    CustomRubric,
    Faithfulness,
    Hallucination,
    Refusal,
)
from jevals.security import PHI, PII, SecretsExposure, SystemPromptLeakage, Toxicity


def test_tool_call_risk_actions(messages):
    s = {"messages": messages, "tool_call": {"name": "refund", "args": {"order_id": "A123", "amount": 100}}}
    hi = MockBackend(
        answers={
            "tool_call_risk.action": {"approve": 0.9, "escalate": 0.05, "block": 0.05},
            "tool_call_risk.destructive": 0.1,
            "tool_call_risk.grounded": 0.9,
        }
    )
    assert evaluate(s, [ToolCallRisk()], backend=hi).tool_call_risk.action == "allow"
    esc = MockBackend(
        answers={
            "tool_call_risk.action": {"approve": 0.9, "escalate": 0.05, "block": 0.05},
            "tool_call_risk.destructive": 0.9,
            "tool_call_risk.grounded": 0.9,
        }
    )
    assert evaluate(s, [ToolCallRisk()], backend=esc).tool_call_risk.action == "escalate"
    blk = MockBackend(answers={"tool_call_risk.action": {"approve": 0.1, "escalate": 0.2, "block": 0.7}})
    r = evaluate(s, [ToolCallRisk()], backend=blk).tool_call_risk
    assert r.action == "block" and r.answer == "block" and "block=0.70" in r.detail
    # state includes tool description for the proposed call
    st = ToolCallRisk().state(Sample(s, tools=[{"name": "refund", "description": "Issue a refund"}]))
    assert st["tool_description"] == "Issue a refund" and st["tool_call"]["tool"] == "refund"


def test_loop_detection_pre_checks(messages):
    r = evaluate({"messages": messages}, [LoopDetection()], backend=MockBackend())
    assert r.loop_detection.passed is True and "fewer than 2" in r.loop_detection.detail
    rep = [
        {"role": "user", "content": "find it"},
        *[
            {
                "role": "assistant",
                "tool_calls": [{"id": f"c{i}", "function": {"name": "search", "arguments": '{"q": "same"}'}}],
            }
            for i in range(3)
        ],
    ]
    be = MockBackend(answers={"loop_detection.stuck": 0.9})
    r = evaluate({"messages": rep}, [LoopDetection(escalate_below=0.5)], backend=be)
    assert r.loop_detection.passed is False and len(be.calls) == 1
    assert LoopDetection(escalate_below=0.5).decide(r.loop_detection, Sample(messages=rep)) == "escalate"


def test_deterministic_trajectory_and_f1(messages):
    s = {"messages": messages, "expected_tool_calls": ["search_orders"]}
    r = evaluate(s, [TrajectoryMatch(mode="strict"), ToolCallF1()], backend=MockBackend())
    assert (
        r.trajectory_match.passed is True and r.tool_call_f1.score == 0.0
    )  # F1 matches args too; expected has {}
    r = evaluate(
        {
            "messages": messages,
            "expected_tool_calls": [{"name": "search_orders", "args": {"email": "jane@example.com"}}],
        },
        [ToolCallF1()],
        backend=MockBackend(),
    )
    assert r.tool_call_f1.score == 1.0
    r = evaluate(
        {"messages": messages, "expected_tool_calls": ["search_orders", "refund"]},
        [TrajectoryMatch(mode="subset"), TrajectoryMatch(mode="superset")],
        backend=MockBackend(),
    )
    assert r.results[0].passed is True and r.results[1].passed is False
    with pytest.raises(ValueError):
        TrajectoryMatch(mode="weird")


def test_score_evals(messages):
    be = MockBackend(answers={"goal_completion.q": {0: 0.0, 1: 0.1, 2: 0.3, 3: 0.6}, "quality.q": 4})
    r = evaluate(
        {"messages": messages},
        [GoalCompletion(), Quality(levels=["unhelpful", "partially", "adequate", "good", "excellent"])],
        backend=be,
    )
    assert r.goal_completion.score == pytest.approx(2.5 / 3) and r.goal_completion.passed
    assert r.quality.score == 1.0 and "excellent" in r.quality.detail


def test_argument_validity_and_step_progress(messages):
    s = {"messages": messages, "tools": [{"name": "search_orders", "parameters": {"x": 1}}]}
    st = ArgumentValidity().state(Sample(s))
    assert st["tool_call"]["tool"] == "search_orders" and st["tool_schema"] == {"x": 1}
    r = evaluate(s, [ArgumentValidity(), StepProgress()], backend=MockBackend(default_noul=0.8))
    assert r.argument_validity.passed and r.step_progress.passed


def test_faithfulness_family():
    s = {
        "input": "Return window?",
        "output": "Pro refunds within 30 days. Free plan is also refundable.",
        "contexts": ["Pro and Team: 30 days. Free plan: no refunds."],
        "expected": "30 days on Pro and Team.",
    }
    be = MockBackend(
        answers={
            "faithfulness.s0": 0.95,
            "faithfulness.s1": 0.05,
            "hallucination.s0": 0.95,
            "hallucination.s1": 0.05,
            "context_recall.r0": 0.9,
            "context_precision.chunk0": 0.9,
        }
    )
    r = evaluate(s, [Faithfulness(), Hallucination(), ContextRecall(), ContextPrecision()], backend=be)
    assert r.faithfulness.score == 0.5 and r.faithfulness.passed is False
    assert r.hallucination.probability == 0.5 and "1/2 statements unsupported" in r.hallucination.detail
    assert r.context_recall.score == 1.0 and r.context_precision.score == 1.0
    assert len(be.calls) == 1  # all four share state keys without conflict


def test_answer_relevancy_and_refusal():
    be = MockBackend(
        answers={
            "answer_relevancy.relevance": 3,
            "answer_relevancy.noncommittal": 0.0,
            "refusal.kind": "over_refusal",
        }
    )
    r = evaluate({"input": "q", "output": "a"}, [AnswerRelevancy(), Refusal()], backend=be)
    assert (
        r.answer_relevancy.score == 1.0 and r.refusal.passed is False and r.refusal.answer == "over_refusal"
    )


def test_custom_rubric():
    ev = CustomRubric("Brand voice?", levels=["off", "close", "on"], name="brand_voice", threshold=0.5)
    r = evaluate({"input": "q", "output": "a"}, [ev], backend=MockBackend(answers={"brand_voice.q": 2}))
    assert r.brand_voice.score == 1.0


def test_pii_redact_and_not_personal():
    text = "Contact support@acme.com or call 555-123-4567."
    ev = PII(surface="output", action="redact")
    r = evaluate({"output": text}, [ev], backend=MockBackend(answers={"pii.personal": 0.1}))
    assert r.pii.passed is True and "not personal" in r.pii.detail
    assert r.pii.value == "Contact <EMAIL_ADDRESS> or call <PHONE_NUMBER>."
    assert ev.decide(r.pii, Sample(output=text)) == "modify"
    r = evaluate({"output": "nothing here"}, [PII()], backend=MockBackend())
    assert r.pii.passed and r.pii.detail == "no entities"


def test_phi_needs_health_context():
    be = MockBackend(answers={"phi.phi": 0.9})
    r = evaluate({"output": "Patient John (MRN: 44812) was diagnosed with diabetes."}, [PHI()], backend=be)
    assert (
        r.phi.passed is False
        and "diagnosed" in r.phi.evidence["terms"]
        and be.calls[0]["state"]["health_terms"]
    )
    r = evaluate({"output": "Order A123 shipped."}, [PHI()], backend=MockBackend())
    assert r.phi.passed is True and not r.phi.answers


def test_secrets_structured_vs_generic():
    # Structured token with a decisive shape: no question asked.
    r = evaluate(
        {"output": "key: AKIAJ7Q2X9LMN4P8R6TB"}, [SecretsExposure(action="redact")], backend=MockBackend()
    )
    assert (
        r.secrets_exposure.passed is False
        and not r.secrets_exposure.answers
        and r.secrets_exposure.value.endswith("R6TB")
    )
    # Structured shape but placeholder-looking (AWS's own docs example): ask.
    be = MockBackend(answers={"secrets_exposure.real": 0.1})
    r = evaluate({"output": "key: AKIAIOSFODNN7EXAMPLE"}, [SecretsExposure()], backend=be)
    assert r.secrets_exposure.passed is True and len(be.calls) == 1
    # Connection string with placeholder credentials: ask.
    be = MockBackend(answers={"secrets_exposure.real": 0.1})
    r = evaluate({"output": "postgres://USER:PASSWORD@host:5432/db"}, [SecretsExposure()], backend=be)
    assert r.secrets_exposure.passed is True and len(be.calls) == 1
    # Connection string with real-looking credentials: still ask (0.7), model says real.
    be = MockBackend(answers={"secrets_exposure.real": 0.95})
    r = evaluate(
        {"output": "postgres://admin:Tr0ub4dor3x@db.internal:5432/prod"}, [SecretsExposure()], backend=be
    )
    assert r.secrets_exposure.passed is False and len(be.calls) == 1
    be = MockBackend(answers={"secrets_exposure.real": 0.1})
    r = evaluate({"output": "password = your-password-here-please"}, [SecretsExposure()], backend=be)
    assert r.secrets_exposure.passed is True and len(be.calls) == 1
    r = evaluate({"output": "all clear"}, [SecretsExposure()], backend=MockBackend())
    assert r.secrets_exposure.passed is True


def test_toxicity_inverts_and_leakage_requires_system_prompt(messages):
    be = MockBackend(answers={"toxicity.q": 3, "system_prompt_leakage.q": 0.9})
    r = evaluate({"messages": messages}, [Toxicity(), SystemPromptLeakage()], backend=be)
    assert r.toxicity.score == 0.0 and r.toxicity.passed is False
    assert r.system_prompt_leakage.passed is False
    r = evaluate({"input": "a", "output": "b"}, [SystemPromptLeakage()], backend=MockBackend())
    assert r.system_prompt_leakage.skipped


def test_mock_json_shapes_are_serializable(sample):
    be = MockBackend()
    evaluate(sample, [ToolCallRisk()], backend=be) if False else None
    r = evaluate({**sample, "tool_call": {"name": "refund", "args": {"a": 1}}}, [ToolCallRisk()], backend=be)
    json.dumps(be.calls[0]["state"], default=str)
    assert r.tool_call_risk.answers
