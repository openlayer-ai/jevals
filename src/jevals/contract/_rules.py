"""An agent's system prompt as a list of rules with stable ids.

The id is a hash of the rule's normalised text, so a rule keeps its id when the prompt is
reordered and gets a new one when it is reworded. That is what lets `check` say which
tests went stale.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, field_validator, model_validator

from .._text import split_sentences
from .._types import Noul, Usage
from ..backends import Backend, resolve
from ..backends._base import run_coro_sync

RULE_ID = re.compile(r"^r-[0-9a-f]{6}$")
_HEADING = re.compile(r"(?m)^[ \t]*#{1,6}[ \t].*$")


def normalize_rule(text: str) -> str:
    s = re.sub(r"\s+", " ", text.strip().lower())
    return s.rstrip(" .!?;:")


def rule_id(text: str) -> str:
    return "r-" + hashlib.sha256(normalize_rule(text).encode()).hexdigest()[:6]


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


class Rule(BaseModel):
    id: str
    text: str

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        if not RULE_ID.match(v):
            raise ValueError(f"rule id {v!r} must look like r-3f2a9c")
        return v


class Contract(BaseModel):
    """`rules` are what the agent must do. `excluded` are lines of the prompt that are not
    rules (dropped by the model, or moved there by hand): `check --prompt` never reports them
    as new."""

    prompt_sha256: str
    source: str | None = None
    rules: list[Rule]
    excluded: list[Rule] = []

    @model_validator(mode="after")
    def _unique(self) -> Contract:
        seen: set[str] = set()
        dupes = sorted({r.id for r in [*self.rules, *self.excluded] if r.id in seen or seen.add(r.id)})
        if dupes:
            raise ValueError(f"rule ids must be unique; repeated: {dupes}")
        return self

    def ids(self) -> set[str]:
        return {r.id for r in self.rules}

    def get(self, rid: str) -> Rule | None:
        return next((r for r in self.rules if r.id == rid), None)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(), sort_keys=False, allow_unicode=True, width=110)

    def dump(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_yaml())
        return p


def load_contract(path: str | Path) -> Contract:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping with prompt_sha256 and rules")
    return Contract.model_validate(data)


def split_rules(prompt: str) -> list[str]:
    """Candidate rules: sentences and list items, headings dropped, duplicates removed."""
    out: list[str] = []
    seen: set[str] = set()
    # Headings go before splitting: a heading and the line under it are one block to the
    # splitter, and dropping the block afterwards would drop that line with it.
    for line in split_sentences(_HEADING.sub("", prompt)):
        key = normalize_rule(line)
        if key and key not in seen:
            seen.add(key)
            out.append(line)
    return out


def prompt_from_trace(path: str | Path, index: int = 0) -> str:
    from .._runner import load_jsonl
    from .._sample import as_sample

    rows = load_jsonl(path)
    if not 0 <= index < len(rows):
        raise ValueError(f"{path} has {len(rows)} traces; no trace at index {index}")
    prompt = as_sample(rows[index]).system_prompt
    if not prompt:
        raise ValueError(f"{path} trace {index} has no system prompt")
    return prompt


_IS_RULE = {
    "true": "An instruction, constraint or prohibition on what the agent does or writes.",
    "false": "Background, a description of the agent's role, an example, or a heading.",
}


class Extraction(BaseModel):
    contract: Contract
    dropped: list[str]
    usage: Usage


async def aextract_rules(
    prompt: str,
    *,
    backend: str | Backend | None = None,
    use_model: bool = True,
    source: str | None = None,
) -> Extraction:
    """Split `prompt` into rules. With `use_model`, one decision-model request drops the lines
    that aren't rules (role text, background, examples); without it every line is kept."""
    lines = split_rules(prompt)
    if not lines:
        raise ValueError("the prompt has no rules: nothing but headings or blank lines")
    keep = [True] * len(lines)
    usage = Usage()
    if use_model:
        questions: dict[str, Any] = {
            f"l{i}": Noul(f"Is lines[{i}] something the agent must or must not do?", criteria=_IS_RULE)
            for i in range(len(lines))
        }
        resp = await resolve(backend).run({"lines": lines}, questions)
        usage = resp.usage
        keep = [resp.answers[f"l{i}"].probability >= 0.5 for i in range(len(lines))]
    rules = [Rule(id=rule_id(t), text=t) for t, k in zip(lines, keep) if k]
    excluded = [Rule(id=rule_id(t), text=t) for t, k in zip(lines, keep) if not k]
    contract = Contract(prompt_sha256=prompt_sha256(prompt), source=source, rules=rules, excluded=excluded)
    return Extraction(contract=contract, dropped=[t for t, k in zip(lines, keep) if not k], usage=usage)


def extract_rules(
    prompt: str,
    *,
    backend: str | Backend | None = None,
    use_model: bool = True,
    source: str | None = None,
) -> Extraction:
    return run_coro_sync(aextract_rules(prompt, backend=backend, use_model=use_model, source=source))
