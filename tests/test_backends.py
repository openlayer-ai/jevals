import json

import httpx
import pytest
import respx

from jevals import Choice, MockBackend, Noul, Score
from jevals.backends import BackendError, resolve
from jevals.backends._wire import (
    answers_from_typesafe,
    answers_from_vercel,
    questions_to_typesafe,
    questions_to_vercel,
)
from jevals.backends.llm import LLMBackend
from jevals.backends.typesafe import TypeSafeBackend
from jevals.backends.vercel import VercelBackend

QS = {
    "a": Noul("yes?"),
    "b": Choice("pick", criteria={"x": "X", "y": "Y"}),
    "c": Score("rate", criteria=["lo", "mid", "hi"]),
}


def test_wire_question_shapes():
    ts = questions_to_typesafe(QS)
    assert (
        ts["a"] == {"type": "noul", "instructions": "yes?"}
        and ts["b"]["criteria"] == {"x": "X", "y": "Y"}
        and ts["c"]["criteria"] == ["lo", "mid", "hi"]
    )
    assert questions_to_vercel(QS)["a"]["type"] == "boolean"


def test_wire_answer_parsing():
    raw = {
        "a": {"noul": 0.8},
        "b": {"choice": "y", "probabilities": {"x": 0.3, "y": 0.7}, "confidence": 0.4},
        "c": {"score": 1.5, "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6}, "confidence": 0.4},
    }
    out = answers_from_typesafe(raw, QS)
    assert (
        out["a"].probability == 0.8
        and out["b"].choice == "y"
        and out["b"].confidence == 0.4
        and out["c"].score == 1.5
        and out["c"].level == 2
    )
    v = answers_from_vercel(
        {
            "a": {"probability": 0.2},
            "b": {"choice": "x", "probabilities": {"x": 0.9, "y": 0.1}},
            "c": {"score": 0.5, "probabilities": {"0": 0.5, "1": 0.5, "2": 0}},
        },
        QS,
        {"b": 0.8},
    )
    assert v["a"].probability == 0.2 and v["b"].confidence == 0.8 and v["c"].levels == ["lo", "mid", "hi"]
    # missing choice -> argmax; missing score -> expectation
    o = answers_from_typesafe(
        {"b": {"probabilities": {"x": 0.2, "y": 0.8}}, "c": {"probabilities": {"0": 0, "1": 0, "2": 1}}}, QS
    )
    assert o["b"].choice == "y" and o["c"].score == 2.0


@respx.mock
async def test_typesafe_backend_roundtrip(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    route = respx.post("https://api.typesafe.ai/v1/systemone").mock(
        return_value=httpx.Response(
            200, json={"answers": {"a": {"noul": 0.9}}, "usage": {"input_tokens": 120}, "latency_ms": 300}
        )
    )
    be = TypeSafeBackend()
    resp = await be.run({"x": 1}, {"a": Noul("q")})
    assert (
        resp.answers["a"].probability == 0.9 and resp.usage.input_tokens == 120 and resp.usage.requests == 1
    )
    assert resp.usage.cost_usd == pytest.approx(120 / 1e6 * 0.042)
    body = json.loads(route.calls[0].request.content)
    assert body["state"] == {"x": 1} and body["questions"]["a"]["type"] == "noul"
    assert route.calls[0].request.headers["authorization"] == "Bearer k"
    respx.post("https://api.typesafe.ai/v1/systemone").mock(return_value=httpx.Response(401, text="nope"))
    with pytest.raises(BackendError):
        await be.run({}, {"a": Noul("q")})


@respx.mock
async def test_vercel_backend_headers_and_parsing(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "vk")
    route = respx.post("https://ai-gateway.vercel.sh/v4/ai/evaluation-model").mock(
        return_value=httpx.Response(
            200,
            json={
                "answers": {
                    "a": {"probability": 0.3},
                    "b": {"choice": "x", "probabilities": {"x": 0.6, "y": 0.4}},
                },
                "usage": {"inputTokens": 50},
                "providerMetadata": {"typesafe": {"confidence": {"b": 0.2}}},
            },
        )
    )
    be = VercelBackend()
    resp = await be.run({"s": 1}, {"a": QS["a"], "b": QS["b"]})
    assert resp.answers["a"].probability == 0.3 and resp.answers["b"].confidence == 0.2
    req = route.calls[0].request
    assert (
        req.headers["ai-model-id"] == "typesafe-ai/jev" and req.headers["ai-gateway-auth-method"] == "api-key"
    )
    body = json.loads(req.content)
    assert (
        body["questions"]["a"]["type"] == "boolean"
        and body["providerOptions"]["gateway"]["zeroDataRetention"] is True
    )


@respx.mock
async def test_llm_backend_emulation(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "ok")
    content = json.dumps({"a": 0.75, "b": {"x": 3, "y": 1}, "c": {"0": 0, "1": 1, "2": 1}})
    respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "```json\n" + content + "\n```"}}],
                "usage": {"prompt_tokens": 400, "completion_tokens": 40, "cost": 0.0003},
            },
        )
    )
    be = LLMBackend(provider="openrouter")
    resp = await be.run({"s": "x"}, QS)
    assert resp.answers["a"].probability == 0.75
    assert resp.answers["b"].choice == "x" and resp.answers["b"].probabilities["x"] == pytest.approx(0.75)
    assert (
        resp.answers["c"].score == pytest.approx(1.5)
        and resp.usage.cost_usd == 0.0003
        and be.name.startswith("llm:")
    )


@respx.mock
async def test_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "k")
    route = respx.post("https://ai-gateway.vercel.sh/v4/ai/evaluation-model").mock(
        side_effect=[
            httpx.Response(429, json={"error": "slow down"}, headers={"retry-after": "0"}),
            httpx.Response(503, text="upstream"),
            httpx.Response(200, json={"answers": {"a": {"probability": 0.9}}, "usage": {"inputTokens": 10}}),
        ]
    )
    be = VercelBackend()
    be.max_retries = 3
    resp = await be.run({"s": 1}, {"a": QS["a"]})
    assert resp.answers["a"].probability == 0.9 and route.call_count == 3


@respx.mock
async def test_no_retry_on_4xx(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "k")
    route = respx.post("https://ai-gateway.vercel.sh/v4/ai/evaluation-model").mock(
        return_value=httpx.Response(401, text="bad key")
    )
    be = VercelBackend()
    with pytest.raises(BackendError) as ei:
        await be.run({"s": 1}, {"a": QS["a"]})
    assert ei.value.status == 401 and not ei.value.retryable and route.call_count == 1


def test_resolver(monkeypatch):
    for k in (
        "JEVALS_BACKEND",
        "TYPESAFE_API_KEY",
        "AI_GATEWAY_API_KEY",
        "KEV_BASE_URL",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "VERCEL_OIDC_TOKEN",
    ):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(BackendError):
        resolve()
    assert isinstance(resolve("mock"), MockBackend)
    mb = MockBackend()
    assert resolve(mb) is mb
    kev = resolve("kev://localhost:8009")
    assert (
        isinstance(kev, TypeSafeBackend)
        and kev.base_url == "http://localhost:8009"
        and kev.name == "kev"
        and kev.price_per_m_input == 0.0
    )
    monkeypatch.setenv("TYPESAFE_API_KEY", "x")
    assert resolve().name == "jev"
    monkeypatch.setenv("JEVALS_BACKEND", "mock")
    assert isinstance(resolve(), MockBackend)
    with pytest.raises(BackendError):
        resolve("zzz:1")


async def test_mock_backend_matching():
    be = MockBackend(answers={"grounded.*": 0.9, "personal": 0.2}, default_noul=0.4)
    resp = await be.run(
        {},
        {
            "grounded.c0": Noul("a"),
            "grounded.c1": Noul("b"),
            "pii.personal": Noul("c"),
            "other.q": Noul("d"),
            "x.k": Choice("p", criteria={"a": "", "b": ""}),
            "x.s": Score("s", criteria=["l", "h"]),
        },
    )
    a = resp.answers
    assert (
        a["grounded.c0"].probability == 0.9
        and a["pii.personal"].probability == 0.2
        and a["other.q"].probability == 0.4
    )
    assert a["x.k"].probabilities == {"a": 0.5, "b": 0.5} and a["x.s"].score == 0.5
    fn = MockBackend(fn=lambda qid, q, st: True)
    assert (await fn.run({}, {"q": Noul("x")})).answers["q"].probability == 0.95
