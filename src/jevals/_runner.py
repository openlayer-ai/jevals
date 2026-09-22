"""evaluate() / aevaluate(): run evals on a sample in as few requests as possible.

Packing: each eval contributes a state dict and a dict of questions. States are
merged by key. If two evals disagree on the value of a key, the later one is
moved to its own request rather than silently overwriting. Question ids are
namespaced as "<eval>.<qid>" on the wire and un-namespaced before reduce().
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ._eval import Eval, Result
from ._sample import Sample, as_sample
from ._types import Answer, Question, Usage
from .backends import Backend, resolve
from .backends._base import run_coro_sync


class EvalReport(BaseModel):
    """Results for one sample. Attribute access by eval name: `r.grounded`."""

    results: list[Result]
    usage: Usage = Field(default_factory=Usage)
    sample: dict[str, Any] | None = None

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        for r in self.results:
            if r.name == name:
                return r
        raise AttributeError(f"no eval named {name!r}; have {[r.name for r in self.results]}")

    def __getitem__(self, name: str) -> Result:
        return self.__getattr__(name)

    def __iter__(self):
        return iter(self.results)

    @property
    def passed(self) -> bool:
        return all(r.passed is not False for r in self.results if not r.skipped)

    @property
    def failures(self) -> list[Result]:
        return [r for r in self.results if r.passed is False]

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": {r.name: r.model_dump(exclude={"answers", "usage"}) for r in self.results},
            "usage": self.usage.model_dump(),
        }

    def table(self) -> str:
        w = max((len(r.name) for r in self.results), default=10) + 2
        lines = []
        for r in self.results:
            if r.skipped:
                lines.append(f"{r.name:<{w}} skipped   {r.detail}")
                continue
            if r.error:
                lines.append(f"{r.name:<{w}} error     {r.error}")
                continue
            if r.answer is not None:
                head = (
                    f"{r.answer:<24} p={r.probability:.2f}"
                    if r.probability is not None
                    else f"{r.answer:<24}"
                )
            elif (
                r.probability is not None
                and r.score is not None
                and r.passed is not None
                and r.evidence.get("per_claim") is None
            ):
                head = f"{'✓' if r.passed else '✗':<24} p={r.probability:.2f}"
            elif r.score is not None:
                head = f"{r.score:<24.2f}"
            else:
                head = f"{'✓' if r.passed else '✗':<24}"
            lines.append(f"{r.name:<{w}} {head:<32} {r.detail}".rstrip())
        lines.append("")
        lines.append(str(self.usage))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.table()


# --------------------------------------------------------------------------- planning


class _Group:
    def __init__(self) -> None:
        self.state: dict[str, Any] = {}
        self.questions: dict[str, Question] = {}
        self.members: list[tuple[Eval, dict[str, Question]]] = []

    def can_take(self, state: dict[str, Any]) -> bool:
        return all(k not in self.state or self.state[k] == v for k, v in state.items())

    def add(self, ev: Eval, state: dict[str, Any], qs: dict[str, Question]) -> None:
        self.state.update(state)
        for qid, q in qs.items():
            self.questions[f"{ev.key}.{qid}"] = q
        self.members.append((ev, qs))


def _plan(evals: Sequence[Eval], s: Sample) -> tuple[list[_Group], list[Result]]:
    groups: list[_Group] = []
    done: list[Result] = []
    for ev in evals:
        ok, why = ev.applicable(s)
        if not ok:
            done.append(Result(name=ev.name, skipped=True, detail=why))
            continue
        try:
            pre = ev.pre(s)
        except Exception as e:  # noqa: BLE001
            done.append(Result(name=ev.name, error=f"pre: {e}"))
            continue
        if pre is not None:
            pre.name = ev.name
            done.append(pre)
            continue
        try:
            state = ev.state(s)
            qs = ev.questions(s)
        except Exception as e:  # noqa: BLE001
            done.append(Result(name=ev.name, error=f"{type(e).__name__}: {e}"))
            continue
        if not qs:
            try:
                r = ev.reduce({}, s)
                r.name = ev.name
                done.append(r)
            except Exception as e:  # noqa: BLE001
                done.append(Result(name=ev.name, error=f"reduce: {e}"))
            continue
        for g in groups:
            if g.can_take(state):
                g.add(ev, state, qs)
                break
        else:
            g = _Group()
            g.add(ev, state, qs)
            groups.append(g)
    return groups, done


async def _run_group(g: _Group, backend: Backend, s: Sample) -> tuple[list[Result], Usage]:
    results: list[Result] = []
    try:
        resp = await backend.run(g.state, g.questions)
    except Exception as e:  # noqa: BLE001
        for ev, _ in g.members:
            results.append(Result(name=ev.name, error=f"{type(e).__name__}: {e}"))
        return results, Usage()
    for ev, qs in g.members:
        answers: dict[str, Answer] = {}
        for qid in qs:
            a = resp.answers.get(f"{ev.key}.{qid}")
            if a is not None:
                answers[qid] = a
        try:
            r = ev.reduce(answers, s)
        except Exception as e:  # noqa: BLE001
            r = Result(error=f"reduce: {type(e).__name__}: {e}")
        r.name = ev.name
        r.answers = answers
        results.append(r)
    return results, resp.usage


async def aevaluate(
    sample: Any,
    evals: Sequence[Eval] | Eval,
    backend: str | Backend | None = None,
    keep_sample: bool = False,
) -> EvalReport:
    """Run evals on one sample. Async."""
    if isinstance(evals, Eval):
        evals = [evals]
    s = as_sample(sample)
    be = resolve(backend)
    groups, done = _plan(evals, s)
    t0 = time.perf_counter()
    outs = await asyncio.gather(*(_run_group(g, be, s) for g in groups))
    usage = Usage()
    results = list(done)
    for rs, u in outs:
        results.extend(rs)
        usage = usage + u
    usage.latency_ms = (time.perf_counter() - t0) * 1000  # wall clock, groups ran in parallel
    if usage.backend is None:
        usage.backend = be.name
    order = {ev.name: i for i, ev in enumerate(evals)}
    results.sort(key=lambda r: order.get(r.name, 999))
    for r in results:
        r.usage = Usage()  # per-eval usage isn't meaningful when packed
    return EvalReport(results=results, usage=usage, sample=dict(s) if keep_sample else None)


def evaluate(
    sample: Any, evals: Sequence[Eval] | Eval, backend: str | Backend | None = None, **kw: Any
) -> EvalReport:
    """Run evals on one sample. Synchronous."""
    return run_coro_sync(aevaluate(sample, evals, backend, **kw))


# --------------------------------------------------------------------------- datasets


class DatasetReport(BaseModel):
    reports: list[EvalReport]
    usage: Usage = Field(default_factory=Usage)
    wall_ms: float = 0.0
    latencies_ms: list[float] = Field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.reports)

    def results_for(self, name: str) -> list[Result]:
        out = []
        for rep in self.reports:
            for r in rep.results:
                if r.name == name and not r.skipped and not r.error:
                    out.append(r)
        return out

    def failures(self, name: str) -> list[tuple[int, Result]]:
        out = []
        for i, rep in enumerate(self.reports):
            for r in rep.results:
                if r.name == name and r.passed is False:
                    out.append((i, r))
        return out

    def summary(self) -> dict[str, dict[str, Any]]:
        names: list[str] = []
        for rep in self.reports:
            for r in rep.results:
                if r.name not in names:
                    names.append(r.name)
        out: dict[str, dict[str, Any]] = {}
        for name in names:
            rs = self.results_for(name)
            scores = [r.score for r in rs if r.score is not None]
            passes = [r.passed for r in rs if r.passed is not None]
            answers = [r.answer for r in rs if r.answer is not None]
            row: dict[str, Any] = {"n": len(rs)}
            if scores:
                row["mean"] = sum(scores) / len(scores)
            if passes:
                row["pass_rate"] = sum(passes) / len(passes)
            if answers:
                counts: dict[str, int] = {}
                for a in answers:
                    counts[a] = counts.get(a, 0) + 1
                row["answers"] = {
                    k: v / len(answers) for k, v in sorted(counts.items(), key=lambda kv: -kv[1])
                }
            errors = sum(1 for rep in self.reports for r in rep.results if r.name == name and r.error)
            if errors:
                row["errors"] = errors
            out[name] = row
        return out

    def table(self) -> str:
        summ = self.summary()
        w = max((len(n) for n in summ), default=10) + 2
        lines = [f"{'':<{w}} {'n':>7}   {'mean':>5}   {'pass':>6}"]
        for name, row in summ.items():
            mean = f"{row['mean']:.2f}" if "mean" in row else "-"
            pr = f"{row['pass_rate'] * 100:.1f}%" if "pass_rate" in row else "-"
            extra = f"   {row['errors']} errors" if "errors" in row else ""
            lines.append(f"{name:<{w}} {row['n']:>7,}   {mean:>5}   {pr:>6}{extra}")
            for k, v in list(row.get("answers", {}).items())[:6]:
                lines.append(f"  {k:<{w + 20}} {v * 100:>5.1f}%")
        lines.append("")
        lat = sorted(self.latencies_ms)
        p50 = lat[len(lat) // 2] if lat else 0
        p95 = lat[int(len(lat) * 0.95)] if lat else 0
        u = self.usage
        if u.cost_usd is None:
            cost = ""
        elif u.cost_usd < 0.01:
            cost = f" · ${u.cost_usd:.4f}"
        else:
            cost = f" · ${u.cost_usd:.2f}"
        tok = f"{u.input_tokens / 1e6:.1f}M" if u.input_tokens >= 1_000_000 else f"{u.input_tokens:,}"
        lines.append(f"{u.requests:,} requests · {tok} tokens{cost} · p50 {p50:.0f}ms · p95 {p95:.0f}ms")
        return "\n".join(lines)

    def to_records(self) -> list[dict[str, Any]]:
        rows = []
        for i, rep in enumerate(self.reports):
            row: dict[str, Any] = {"index": i}
            for r in rep.results:
                row[f"{r.name}.score"] = r.score
                row[f"{r.name}.passed"] = r.passed
                if r.answer is not None:
                    row[f"{r.name}.answer"] = r.answer
                if r.probability is not None:
                    row[f"{r.name}.p"] = r.probability
            rows.append(row)
        return rows

    def to_pandas(self):
        import pandas as pd  # optional

        return pd.DataFrame(self.to_records())

    def write_jsonl(self, path: str | Path) -> None:
        with open(path, "w") as f:
            for i, rep in enumerate(self.reports):
                f.write(json.dumps({"index": i, **rep.to_dict()}, default=str) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


async def aevaluate_dataset(
    data: str | Path | Iterable[Any],
    evals: Sequence[Eval],
    backend: str | Backend | None = None,
    concurrency: int = 16,
    on_result: Any = None,
) -> DatasetReport:
    rows = load_jsonl(data) if isinstance(data, (str, Path)) else list(data)
    be = resolve(backend)
    sem = asyncio.Semaphore(concurrency)
    lat: list[float] = [0.0] * len(rows)
    reports: list[EvalReport | None] = [None] * len(rows)

    async def one(i: int, row: Any) -> None:
        async with sem:
            t0 = time.perf_counter()
            rep = await aevaluate(row, evals, be)
            lat[i] = (time.perf_counter() - t0) * 1000
            reports[i] = rep
            if on_result:
                on_result(i, rep)

    t0 = time.perf_counter()
    await asyncio.gather(*(one(i, r) for i, r in enumerate(rows)))
    usage = Usage()
    for rep in reports:
        if rep:
            usage = usage + rep.usage
    usage.latency_ms = sum(lat)
    return DatasetReport(
        reports=[r for r in reports if r],
        usage=usage,
        wall_ms=(time.perf_counter() - t0) * 1000,
        latencies_ms=lat,
    )


def evaluate_dataset(
    data: Any, evals: Sequence[Eval], backend: str | Backend | None = None, **kw: Any
) -> DatasetReport:
    return run_coro_sync(aevaluate_dataset(data, evals, backend, **kw))
