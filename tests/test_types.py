import pytest
from pydantic import ValidationError

from jevals import Choice, ChoiceAnswer, Noul, NoulAnswer, Score, ScoreAnswer, Usage
from jevals._types import question_from_dict


def test_noul_positional_and_criteria():
    q = Noul("Is it good?", criteria={"true": "yes", "false": "no"})
    assert q.type == "noul" and q.instructions == "Is it good?" and q.criteria["true"] == "yes"


def test_choice_options_alias_and_bounds():
    q = Choice("Pick", options={"a": "A", "b": "B"})
    assert list(q.criteria) == ["a", "b"]
    with pytest.raises(ValidationError):
        Choice("Pick", criteria={})


def test_score_levels_alias_and_bounds():
    q = Score("Rate", levels=["bad", "ok", "good"])
    assert q.criteria == ["bad", "ok", "good"]
    with pytest.raises(ValidationError):
        Score("Rate", criteria=["one"])


def test_question_from_dict_accepts_boolean_alias():
    q = question_from_dict({"type": "boolean", "instructions": "x"})
    assert isinstance(q, Noul)
    with pytest.raises(ValueError):
        question_from_dict({"type": "nope", "instructions": "x"})


def test_answers():
    a = NoulAnswer(probability=0.7)
    assert a.value is True and float(a) == 0.7
    c = ChoiceAnswer(choice="a", probabilities={"a": 0.7, "b": 0.3})
    assert c.probability == 0.7 and c.p("b") == 0.3 and c.p("zzz") == 0.0
    assert c.confidence == pytest.approx((0.7 - 0.5) / 0.5)
    s = ScoreAnswer(score=2.4, probabilities={0: 0.0, 1: 0.1, 2: 0.4, 3: 0.5}, levels=["a", "b", "c", "d"])
    assert s.level == 3 and s.label == "d" and s.normalized == pytest.approx(0.8)


def test_usage_add_and_str():
    u = Usage(requests=1, input_tokens=1000, cost_usd=0.00004, latency_ms=400) + Usage(
        requests=1, input_tokens=500, latency_ms=100
    )
    assert u.requests == 2 and u.input_tokens == 1500 and u.cost_usd == pytest.approx(0.00004)
    assert "2 requests" in str(u) and "1,500 tokens" in str(u)
