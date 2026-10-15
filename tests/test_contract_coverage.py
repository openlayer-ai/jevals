from jevals import MockBackend
from jevals._declarative import DeclarativeEval
from jevals.contract import Contract, Rule, check_coverage, rule_id

TEXTS = [
    "Ask for the order number when the customer has not given one.",
    "Never promise a refund before checking that the order is inside the 30-day refund window.",
    "Reply in the customer's language.",
]
ASK, REFUND, LANG = (rule_id(t) for t in TEXTS)
CONTRACT = Contract(prompt_sha256="0" * 64, rules=[Rule(id=rule_id(t), text=t) for t in TEXTS])


def ev(name, questions, **top):
    return DeclarativeEval({"name": name, "questions": questions, **top})


def q(text, covers=None, why=None):
    d = {"type": "noul", "instructions": text}
    if covers is not None:
        d["covers"] = covers
    if why:
        d["why"] = why
    return d


def test_clean_contract():
    evals = [
        ev(
            "support",
            {
                "a": q("Asks for the order number?", [ASK]),
                "b": q("Promises a refund?", [REFUND]),
                "c": q("Same language?", [LANG]),
            },
        )
    ]
    rep = check_coverage(CONTRACT, evals, use_model=False)
    assert rep.uncovered == [] and rep.stale == [] and rep.uncited == [] and not rep.failed(strict=True)


def test_uncovered_rule_fails():
    rep = check_coverage(CONTRACT, [ev("s", {"a": q("?", [ASK, REFUND])})], use_model=False)
    assert [r.id for r in rep.uncovered] == [LANG] and rep.failed()


def test_stale_citation_fails():
    rep = check_coverage(CONTRACT, [ev("s", {"a": q("?", [ASK, REFUND, LANG, "r-000000"])})], use_model=False)
    assert [(c.question, c.rule_id) for c in rep.stale] == [("s.a", "r-000000")] and rep.failed()


def test_uncited_fails_only_when_strict_and_observed_counts_as_cited():
    evals = [
        ev("s", {"a": q("?", [ASK, REFUND, LANG]), "b": q("Apologises?", ["observed"], why="tone")}),
        ev("plain", {"x": q("Anything?")}),
    ]
    rep = check_coverage(CONTRACT, evals, use_model=False)
    assert rep.uncited == ["plain.x"]
    assert not rep.failed() and rep.failed(strict=True)


def test_changed_prompt_lists_added_and_removed_rules_and_who_cites_them():
    new_prompt = "- " + TEXTS[0] + "\n- " + TEXTS[2] + "\n- Never share internal ticket ids.\n"
    evals = [ev("s", {"a": q("?", [ASK]), "b": q("?", [REFUND]), "c": q("?", [LANG])})]
    rep = check_coverage(CONTRACT, evals, use_model=False, prompt=new_prompt)
    d = rep.prompt_diff
    assert [r.text for r in d.added] == ["Never share internal ticket ids."]
    assert [r.id for r in d.removed] == [REFUND]
    assert [(c.question, c.rule_id) for c in d.cited_removed] == [("s.b", REFUND)]
    assert rep.failed()


def test_same_prompt_has_no_diff():
    prompt = "\n".join("- " + t for t in TEXTS)
    evals = [ev("s", {"a": q("?", [ASK, REFUND, LANG])})]
    rep = check_coverage(CONTRACT, evals, use_model=False, prompt=prompt)
    assert rep.prompt_diff is not None and not rep.prompt_diff.added and not rep.prompt_diff.removed
    assert not rep.failed()


def test_table_names_each_finding():
    rep = check_coverage(CONTRACT, [ev("s", {"a": q("?", [ASK, "r-000000"])})], use_model=False)
    t = rep.table()
    assert "uncovered" in t and LANG in t and "stale" in t and "s.a" in t


def test_model_is_not_called_when_use_model_is_false():
    mock = MockBackend()
    check_coverage(CONTRACT, [ev("s", {"a": q("?", [ASK])})], backend=mock, use_model=False, prompt=TEXTS[0])
    assert mock.calls == []


def test_model_checks_each_cited_pair_in_one_request_and_flags_doubtful_ones():
    evals = [
        ev(
            "s",
            {
                "a": q("Asks for the order number?", [ASK]),
                "b": q("Is the reply polite?", [REFUND]),
                "c": q("Same language?", [LANG, "observed"], why="x"),
            },
        )
    ]
    mock = MockBackend(answers={"p1": 0.2}, default_noul=0.9)  # p1 = (s.b, REFUND)
    rep = check_coverage(CONTRACT, evals, backend=mock)
    assert [(d.question, d.rule_id, d.p) for d in rep.doubtful] == [("s.b", REFUND, 0.2)]
    assert len(mock.calls) == 1
    state = mock.calls[0]["state"]
    assert set(state["rules"]) == {ASK, REFUND, LANG}
    assert state["questions"]["s.b"] == "Is the reply polite?"
    assert set(mock.calls[0]["questions"]) == {"p0", "p1", "p2"}  # observed is not checked
    assert not rep.failed() and rep.failed(strict=True)
    assert rep.usage.requests == 1


def test_stale_citations_are_not_sent_to_the_model_and_no_pairs_means_no_request():
    mock = MockBackend()
    check_coverage(CONTRACT, [ev("s", {"a": q("?", ["r-000000"])})], backend=mock)
    assert mock.calls == []


# --- final-review fixes ---------------------------------------------------------------------------


def test_unchanged_prompt_never_reports_a_diff_even_after_editing_the_contract():
    from jevals.contract import prompt_sha256

    prompt = "- Role text the person removed by hand.\n" + "\n".join("- " + t for t in TEXTS)
    edited = Contract(prompt_sha256=prompt_sha256(prompt), rules=CONTRACT.rules)  # dropped line 1
    evals = [ev("s", {"a": q("?", [ASK, REFUND, LANG])})]
    rep = check_coverage(edited, evals, use_model=False, prompt=prompt)
    assert rep.prompt_diff is None and not rep.failed()


def test_excluded_rules_are_not_new_when_the_prompt_changes():
    excluded_text = "You are the support agent."
    contract = Contract(
        prompt_sha256="0" * 64,
        rules=CONTRACT.rules,
        excluded=[Rule(id=rule_id(excluded_text), text=excluded_text)],
    )
    prompt = "- " + excluded_text + "\n" + "\n".join("- " + t for t in TEXTS) + "\n- Be brief."
    rep = check_coverage(
        contract, [ev("s", {"a": q("?", [ASK, REFUND, LANG])})], use_model=False, prompt=prompt
    )
    assert [r.text for r in rep.prompt_diff.added] == ["Be brief."]


def test_duplicate_eval_names_are_an_error():
    import pytest

    evals = [ev("same", {"q": q("FIRST?", [ASK])}), ev("same", {"q": q("SECOND?", [REFUND])})]
    with pytest.raises(ValueError, match="same"):
        check_coverage(CONTRACT, evals, use_model=False)


def test_dotted_eval_name_sends_the_real_question_text():
    mock = MockBackend(default_noul=0.9)
    check_coverage(CONTRACT, [ev("support.v2", {"q": q("Asks for the order number?", [ASK])})], backend=mock)
    assert mock.calls[0]["state"]["questions"] == {"support.v2.q": "Asks for the order number?"}


def test_rule_text_edited_under_its_old_id_fails():
    reworded = "Ask for the client number when the customer has not given one."
    contract = Contract(
        prompt_sha256="0" * 64,
        rules=[Rule(id=ASK, text=reworded), *CONTRACT.rules[1:]],
    )
    evals = [ev("s", {"a": q("Asks for the order number?", [ASK]), "b": q("?", [REFUND, LANG])})]
    rep = check_coverage(contract, evals, use_model=False)
    assert [(r.id, r.text) for r in rep.edited] == [(ASK, reworded)]
    assert rep.failed()
    table = rep.table()
    assert f"edited      {ASK}" in table and rule_id(reworded) in table and "cited by s.a" in table


def test_case_and_punctuation_edits_keep_the_id():
    contract = Contract(
        prompt_sha256="0" * 64,
        rules=[
            Rule(id=ASK, text="ask for the ORDER number when the customer has not given one"),
            *CONTRACT.rules[1:],
        ],
    )
    rep = check_coverage(contract, [ev("s", {"a": q("?", [ASK, REFUND, LANG])})], use_model=False)
    assert rep.edited == [] and not rep.failed()
