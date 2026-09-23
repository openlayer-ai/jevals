import json

import pytest

from jevals import MockBackend, split_sentences
from jevals._calibrate import calibrate
from jevals.cli import main
from jevals.security import detect_phi, detect_pii, detect_secrets, redact
from jevals.security._detect import cpf_ok, luhn_ok, npi_ok
from jevals.security._evals import PromptInjection


def test_split_sentences():
    assert split_sentences("Dr. Smith arrived at 5 p.m. today. He left early. Then Mrs. Jones came in!") == [
        "Dr. Smith arrived at 5 p.m. today.",
        "He left early.",
        "Then Mrs. Jones came in!",
    ]
    assert split_sentences("Items:\n- first item here\n- second item here") == [
        "Items:",
        "first item here",
        "second item here",
    ]
    assert split_sentences("") == [] and split_sentences("short") == ["short"]


def test_checksums():
    assert luhn_ok("4111 1111 1111 1111") and not luhn_ok("4111 1111 1111 1112")
    assert npi_ok("1234567893") and not npi_ok("1234567890")
    assert cpf_ok("529.982.247-25") and not cpf_ok("111.111.111-11")


def test_detectors_and_redact():
    text = (
        "Email jane@example.com, SSN 123-45-6789, card 4111 1111 1111 1111, ip 10.0.0.1, CPF 529.982.247-25"
    )
    types = {e["type"] for e in detect_pii(text, use_presidio=False)}
    assert {"EMAIL_ADDRESS", "US_SSN", "CREDIT_CARD", "IP_ADDRESS", "BR_CPF"} <= types
    red = redact(text, detect_pii(text, use_presidio=False))
    assert "jane@example.com" not in red and "<US_SSN>" in red
    ents, terms = detect_phi("Patient MRN: A88213 diagnosed with asthma, NPI 1234567893", use_presidio=False)
    assert (
        {"MEDICAL_RECORD_NUMBER", "US_NPI"} <= {e["type"] for e in ents}
        and "asthma" in terms
        and "diagnosed" in terms
    )
    sec = detect_secrets(
        "token ghp_" + "a" * 36 + " and sk-ant-" + "b" * 24 + " and -----BEGIN RSA PRIVATE KEY-----"
    )
    assert {"GITHUB_TOKEN", "ANTHROPIC_KEY", "PRIVATE_KEY"} <= {e["type"] for e in sec}
    assert redact(
        "card 4111111111111111", detect_pii("card 4111111111111111", use_presidio=False), style="partial"
    ).endswith("1111")


def test_pii_union_covers_presidio_misses(monkeypatch):
    """detect_pii must not drop entities Presidio misses at its own threshold boundary.

    Regression: the regex floor used to get filtered down to only BR_CPF once
    Presidio ran, so anything Presidio missed (a phone number scoring just
    under the cutoff, an SSN it never flagged) silently disappeared instead
    of falling back to the regex scan.
    """
    text = "Contact support@acme.com or call 555-123-4567, ssn 123-45-6789."
    start = text.index("support@acme.com")
    presidio_result = [
        {
            "type": "EMAIL_ADDRESS",
            "text": "support@acme.com",
            "start": start,
            "end": start + len("support@acme.com"),
            "score": 1.0,
        }
    ]
    monkeypatch.setattr(
        "jevals.security._detect._presidio_scan",
        lambda text, entities, threshold: presidio_result,
    )
    types = {e["type"] for e in detect_pii(text, use_presidio=True)}
    assert {"EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN"} <= types


def test_pii_restricts_presidio_to_curated_entity_types(monkeypatch):
    """Presidio's full NER surface must not leak into PII detection.

    Regression: detect_pii asked Presidio for every entity type it supports
    (entities=None), including ones this project never modeled as PII, like
    DATE_TIME ("Tuesday" in a shipping update) or PERSON. Only types that
    have a regex counterpart in _PII_PATTERNS should ever be requested.
    """
    captured = {}

    def fake_scan(text, entities, threshold):
        captured["entities"] = entities
        return []

    monkeypatch.setattr("jevals.security._detect._presidio_scan", fake_scan)
    detect_pii("Your order shipped and should arrive Tuesday.", use_presidio=True)
    assert captured["entities"] is not None
    assert "DATE_TIME" not in captured["entities"]
    assert "PERSON" not in captured["entities"]
    assert {"EMAIL_ADDRESS", "US_SSN", "CREDIT_CARD", "PHONE_NUMBER"} <= set(captured["entities"])


def test_calibrate_threshold_table():
    rows = []
    for i in range(40):
        good = i % 2 == 0
        rows.append(
            {
                "messages": [{"role": "user", "content": f"msg {i}"}],
                "label": "pass" if good else "fail",
                "_p": 0.85 if good else 0.2,
            }
        )
    # backend answers p(injection) low for good rows, high for bad; PromptInjection score = 1 - p
    be = MockBackend(
        fn=lambda qid, q, st: (
            0.1
            if "msg" in json.dumps(st)
            and int(json.dumps(st).split("msg ")[1].split('"')[0].split()[0]) % 2 == 0
            else 0.9
        )
    )
    cal = calibrate(rows, PromptInjection(), "label", backend=be)
    assert cal.n == 40 and cal.auroc == 1.0 and cal.brier < 0.05
    row = next(r for r in cal.rows if abs(r.threshold - 0.5) < 1e-9)
    assert row.auto_rate == 0.5 and row.false_pass == 0.0 and row.false_fail == 0.0
    assert "Brier" in cal.table() and cal.best(0.01).threshold >= 0.5


def test_cli_commands(tmp_path, capsys, sample):
    assert main(["list"]) == 0 and "tool_call_risk" in capsys.readouterr().out
    assert (
        main(["list", "--json", "--category", "security"]) == 0
        and "prompt_injection" in capsys.readouterr().out
    )
    assert main(["schema"]) == 0 and json.loads(capsys.readouterr().out)["title"] == "jevals eval"
    assert main(["docs", "--llm"]) == 0 and "## Gate" in capsys.readouterr().out
    assert main(["describe", "grounded"]) == 0 and "claims" in capsys.readouterr().out

    p = tmp_path / "traces.jsonl"
    p.write_text("\n".join(json.dumps(sample) for _ in range(3)) + "\n")
    out = tmp_path / "out.jsonl"
    assert (
        main(
            [
                "--backend",
                "mock",
                "run",
                str(p),
                "--evals",
                "agent.tool_choice,agent.grounded,security.indirect_injection",
                "--out",
                str(out),
                "--quiet",
            ]
        )
        == 0
    )
    text = capsys.readouterr().out
    assert "tool_choice" in text and "3 requests" in text and len(out.read_text().splitlines()) == 3

    sp = tmp_path / "s.json"
    sp.write_text(json.dumps({**sample, "tool_call": {"name": "refund", "args": {"order_id": "A123"}}}))
    code = main(["--backend", "mock", "gate", str(sp), "--evals", "agent.tool_call_risk", "--json"])
    d = json.loads(capsys.readouterr().out)
    assert d["action"] == "escalate" and code == 2  # uniform mock -> escalate

    bad = tmp_path / "bad.yaml"
    bad.write_text("name: nope\n")
    assert main(["validate", str(bad)]) == 1 and "FAIL" in capsys.readouterr().out

    with pytest.raises(SystemExit):
        main(["run", str(p)])  # --evals required
