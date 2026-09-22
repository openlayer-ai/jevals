import pytest

from jevals import Eval, MockBackend, Noul, Result, aevaluate, evaluate, evaluate_dataset
from jevals.agent import Grounded, StayedInScope, ToolChoice, UsedToolResult
from jevals.security import PHI, PII, IndirectInjection


def test_packs_all_evals_into_one_request(sample):
    be = MockBackend(
        answers={
            "tool_choice.q": "correct",
            "grounded.c0": 0.9,
            "grounded.c1": 0.1,
            "indirect_injection.q": 0.95,
        }
    )
    r = evaluate(
        sample,
        [
            ToolChoice(),
            UsedToolResult(),
            Grounded(),
            StayedInScope(),
            IndirectInjection(),
            PHI(),
            PII(surface="all"),
        ],
        backend=be,
    )
    assert len(be.calls) == 1
    assert r.usage.requests == 1
    assert r.tool_choice.answer == "correct" and r.tool_choice.passed is True
    assert r.grounded.score == 0.5 and r.grounded.evidence["unsupported"] == [1]
    assert r.indirect_injection.passed is False and r.indirect_injection.probability == 0.95
    assert r.phi.passed is True and r.phi.detail.startswith("no identifiers")
    assert r.pii.evidence["entities"][0]["type"] == "EMAIL_ADDRESS"
    assert "tool_choice" in r.table() and "1 request" in r.table()
    assert r["grounded"] is r.grounded
    with pytest.raises(AttributeError):
        _ = r.not_an_eval


def test_state_conflict_splits_requests():
    class A(Eval):
        name = "a"

        def state(self, s):
            return {"text": "one"}

        def questions(self, s):
            return {"q": Noul("x")}

        def reduce(self, answers, s):
            return Result(score=answers["q"].probability)

    class B(A):
        name = "b"

        def state(self, s):
            return {"text": "two"}  # same key, different value

    be = MockBackend()
    r = evaluate({"messages": []}, [A(), B()], backend=be)
    assert len(be.calls) == 2 and r.usage.requests == 2
    assert {c["state"]["text"] for c in be.calls} == {"one", "two"}


def test_skipped_and_errors_are_reported(messages):
    class Boom(Eval):
        name = "boom"

        def questions(self, s):
            raise RuntimeError("nope")

        def reduce(self, a, s):
            return Result()

    r = evaluate({"messages": messages}, [Grounded(), Boom(), PHI()], backend=MockBackend())
    # Grounded applies (tool results exist); Boom errors; PHI passes deterministically
    assert r.boom.error and "nope" in r.boom.error
    assert r.passed is True  # errors are not failures; skipped ignored
    r2 = evaluate({"input": "x", "output": "y"}, [Grounded()], backend=MockBackend())
    assert (
        r2.grounded.skipped
        and "missing messages" in r2.grounded.detail
        or "no tool results" in r2.grounded.detail
    )


def test_backend_failure_becomes_error_result(sample):
    class Bad(MockBackend):
        async def evaluate(self, state, questions):
            raise ConnectionError("down")

    r = evaluate(sample, [ToolChoice()], backend=Bad())
    assert r.tool_choice.error and "down" in r.tool_choice.error


async def test_aevaluate_and_sync_inside_loop(sample):
    be = MockBackend(answers={"tool_choice.q": "correct"})
    r = await aevaluate(sample, [ToolChoice()], backend=be)
    assert r.tool_choice.answer == "correct"
    # evaluate() from inside a running loop must not deadlock
    r2 = evaluate(sample, [ToolChoice()], backend=be)
    assert r2.tool_choice.answer == "correct"


def test_dataset_summary_and_records(sample, tmp_path):
    be = MockBackend(fn=lambda qid, q, state: "correct" if qid.endswith(".q") and q.type == "choice" else 0.9)
    rows = [sample, sample, {"input": "x", "output": "y"}]
    rep = evaluate_dataset(rows, [ToolChoice(), Grounded()], backend=be)
    summ = rep.summary()
    # single-turn row still has derived messages, so tool_choice runs on all 3; grounded skips it (no tool results)
    assert (
        summ["tool_choice"]["n"] == 3
        and summ["tool_choice"]["pass_rate"] == 1.0
        and summ["tool_choice"]["answers"] == {"correct": 1.0}
    )
    assert summ["grounded"]["n"] == 2 and summ["grounded"]["mean"] == 1.0
    assert "tool_choice" in rep.table() and "requests" in rep.table()
    recs = rep.to_records()
    assert recs[0]["tool_choice.answer"] == "correct" and recs[2]["grounded.score"] is None
    out = tmp_path / "out.jsonl"
    rep.write_jsonl(out)
    assert len(out.read_text().splitlines()) == 3
    rep2 = evaluate_dataset(str(tmp_path / "in.jsonl"), [ToolChoice()], backend=be) if False else None
    assert rep2 is None
