"""Which prompt rules have a question, and which questions still match the prompt."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from .._declarative import DeclarativeEval
from .._types import Noul, Usage
from ..backends import Backend, resolve
from ..backends._base import run_coro_sync
from ._rules import Contract, Rule, aextract_rules, prompt_sha256, rule_id, split_rules

OBSERVED = "observed"


class Citation(BaseModel):
    question: str  # "<eval>.<qid>"
    rule_id: str
    instructions: str = ""


class Doubt(BaseModel):
    question: str
    rule_id: str
    p: float


class PromptDiff(BaseModel):
    added: list[Rule]
    removed: list[Rule]
    cited_removed: list[Citation]


class CoverageReport(BaseModel):
    uncovered: list[Rule] = []
    stale: list[Citation] = []
    # Rules whose text was changed by hand under the old id. The id is the text's hash, so
    # a reworded rule must get a new one, and the questions citing it must be looked at again.
    edited: list[Rule] = []
    edited_cited: list[Citation] = []
    prompt_diff: PromptDiff | None = None
    uncited: list[str] = []
    doubtful: list[Doubt] = []
    usage: Usage = Field(default_factory=Usage)

    def failed(self, strict: bool = False) -> bool:
        changed = self.prompt_diff is not None and bool(self.prompt_diff.added or self.prompt_diff.removed)
        hard = bool(self.uncovered or self.stale or self.edited or changed)
        return hard or (strict and bool(self.uncited or self.doubtful))

    def table(self) -> str:
        rows: list[str] = []
        for r in self.uncovered:
            rows.append(f"uncovered   {r.id}  {r.text}")
        for c in self.stale:
            rows.append(f"stale       {c.question} cites {c.rule_id}, which the contract does not have")
        for r in self.edited:
            rows.append(f"edited      {r.id}  {r.text}  (text changed; its id is now {rule_id(r.text)})")
            rows.extend(
                f"              cited by {c.question}" for c in self.edited_cited if c.rule_id == r.id
            )
        if self.prompt_diff:
            for r in self.prompt_diff.added:
                rows.append(f"new rule    {r.id}  {r.text}  (not in the contract)")
            for r in self.prompt_diff.removed:
                rows.append(f"gone rule   {r.id}  {r.text}  (no longer in the prompt)")
            for c in self.prompt_diff.cited_removed:
                rows.append(f"              cited by {c.question}")
        for qid in self.uncited:
            rows.append(f"uncited     {qid}  (no covers)")
        for d in self.doubtful:
            rows.append(f"doubtful    {d.question} -> {d.rule_id}  p={d.p:.2f}")
        head = "contract coverage: clean" if not rows else f"contract coverage: {len(rows)} finding(s)"
        tail = f"\n{self.usage}" if self.usage.requests else ""
        return "\n".join([head, *rows]) + tail


def _citations(evals: Sequence[DeclarativeEval]) -> tuple[list[Citation], list[str]]:
    names = [ev.name for ev in evals]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"eval names must be unique to check coverage; repeated: {dupes}")
    cites: list[Citation] = []
    uncited: list[str] = []
    for ev in evals:
        for qid, ids in getattr(ev, "covers", {}).items():
            name = f"{ev.name}.{qid}"
            if not ids:
                uncited.append(name)
            ins = ev._questions[qid].instructions
            text = ins if isinstance(ins, str) else str(ins)
            cites.extend(Citation(question=name, rule_id=i, instructions=text) for i in ids if i != OBSERVED)
    return cites, uncited


async def acheck_coverage(
    contract: Contract,
    evals: Sequence[DeclarativeEval],
    *,
    backend: str | Backend | None = None,
    use_model: bool = True,
    prompt: str | None = None,
) -> CoverageReport:
    cites, uncited = _citations(evals)
    known = contract.ids()
    cited_ids = {c.rule_id for c in cites}
    rep = CoverageReport(
        uncovered=[r for r in contract.rules if r.id not in cited_ids],
        stale=[c for c in cites if c.rule_id not in known],
        uncited=uncited,
    )
    rep.edited = [r for r in contract.rules if r.id != rule_id(r.text)]
    edited_ids = {r.id for r in rep.edited}
    rep.edited_cited = [c for c in cites if c.rule_id in edited_ids]
    if prompt is not None and prompt_sha256(prompt) != contract.prompt_sha256:
        # The prompt changed. A rule is gone when its text is no longer in the prompt at all,
        # and new when the model calls a line a rule the contract neither has nor excludes.
        ex = await aextract_rules(prompt, backend=backend, use_model=use_model)
        in_prompt = {rule_id(t) for t in split_rules(prompt)}
        known_lines = known | {r.id for r in contract.excluded}
        removed = [r for r in contract.rules if r.id not in in_prompt]
        rep.prompt_diff = PromptDiff(
            added=[r for r in ex.contract.rules if r.id not in known_lines],
            removed=removed,
            cited_removed=[c for c in cites if c.rule_id in {r.id for r in removed}],
        )
        rep.usage = rep.usage + ex.usage
    pairs = [c for c in cites if c.rule_id in known]
    if use_model and pairs:
        state = {
            "rules": {rid: contract.get(rid).text for rid in sorted({c.rule_id for c in pairs})},
            "questions": {c.question: c.instructions for c in sorted(pairs, key=lambda c: c.question)},
        }
        questions = {
            f"p{i}": Noul(
                f"Does questions[{c.question!r}] test whether the agent follows rules[{c.rule_id!r}]?"
            )
            for i, c in enumerate(pairs)
        }
        resp = await resolve(backend).run(state, questions)
        rep.usage = rep.usage + resp.usage
        for i, c in enumerate(pairs):
            p = resp.answers[f"p{i}"].probability
            if p < 0.5:
                rep.doubtful.append(Doubt(question=c.question, rule_id=c.rule_id, p=p))
    return rep


def check_coverage(
    contract: Contract,
    evals: Sequence[DeclarativeEval],
    *,
    backend: str | Backend | None = None,
    use_model: bool = True,
    prompt: str | None = None,
) -> CoverageReport:
    return run_coro_sync(
        acheck_coverage(contract, evals, backend=backend, use_model=use_model, prompt=prompt)
    )
