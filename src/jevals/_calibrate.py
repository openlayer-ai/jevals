"""Calibration: pick a threshold on your labels, and report how well probabilities match outcomes.

    jevals calibrate labeled.jsonl --eval evals/tool_call_risk.yaml --label human_decision

Each row is a sample plus a label field. For a pass/fail eval the label is a bool (or "pass"/"fail").
For a choice-shaped eval (tool_call_risk) the label is the option a human would have picked;
"approve" counts as positive.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from ._eval import Eval
from ._runner import DatasetReport, aevaluate_dataset
from .backends import Backend
from .backends._base import run_coro_sync

_POSITIVE = {"true", "pass", "passed", "ok", "allow", "approve", "yes", "1", "good", "safe"}


def _label_to_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return v >= 0.5
    return str(v).strip().lower() in _POSITIVE


class ThresholdRow(BaseModel):
    threshold: float
    auto_rate: float  # share of samples the eval would pass / approve at this threshold
    false_pass: float  # passed but label says no  (wrong approvals)
    false_fail: float  # failed but label says yes (missed approvals / over-escalation)
    precision: float
    recall: float


class Calibration(BaseModel):
    n: int
    brier: float
    ece: float
    auroc: float | None
    rows: list[ThresholdRow]
    current_threshold: float | None = None
    bins: list[dict[str, float]] = Field(default_factory=list)

    def best(self, max_false_pass: float = 0.01) -> ThresholdRow | None:
        ok = [r for r in self.rows if r.false_pass <= max_false_pass]
        return max(ok, key=lambda r: r.auto_rate) if ok else None

    def table(self) -> str:
        lines = [f"{'threshold':<10} {'auto-pass':>10} {'wrong passes':>13} {'missed passes':>14}"]
        for r in self.rows:
            mark = (
                "   current"
                if self.current_threshold is not None and abs(r.threshold - self.current_threshold) < 1e-9
                else ""
            )
            lines.append(
                f"{r.threshold:<10.2f} {r.auto_rate * 100:>9.1f}% {r.false_pass * 100:>12.1f}% {r.false_fail * 100:>13.1f}%{mark}"
            )
        auroc = f" · AUROC {self.auroc:.3f}" if self.auroc is not None else ""
        lines.append(f"Brier {self.brier:.3f} · ECE {self.ece:.3f}{auroc} · n={self.n:,}")
        return "\n".join(lines)


def _headline(r: Any) -> float | None:
    """The 0..1 value we threshold: score (higher = passes)."""
    return r.score if r.score is not None else r.probability


def calibrate_from_report(
    report: DatasetReport,
    eval_name: str,
    labels: Sequence[Any],
    thresholds: Sequence[float] | None = None,
    current: float | None = None,
) -> Calibration:
    ps: list[float] = []
    ys: list[bool] = []
    for rep, lab in zip(report.reports, labels):
        y = _label_to_bool(lab)
        if y is None:
            continue
        for r in rep.results:
            if r.name == eval_name and not r.skipped and not r.error:
                v = _headline(r)
                if v is not None:
                    ps.append(float(v))
                    ys.append(y)
    n = len(ps)
    if n == 0:
        raise ValueError(f"no scored results for {eval_name!r} with labels")
    brier = sum((p - (1.0 if y else 0.0)) ** 2 for p, y in zip(ps, ys)) / n
    # ECE with 10 equal-width bins
    bins: list[dict[str, float]] = []
    ece = 0.0
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        idx = [i for i, p in enumerate(ps) if lo <= p < hi or (b == 9 and p == 1.0)]
        if not idx:
            continue
        conf = sum(ps[i] for i in idx) / len(idx)
        acc = sum(1.0 for i in idx if ys[i]) / len(idx)
        ece += len(idx) / n * abs(conf - acc)
        bins.append({"lo": lo, "hi": hi, "n": len(idx), "confidence": conf, "accuracy": acc})
    auroc = _auroc(ps, ys)
    ths = list(thresholds) if thresholds else [round(x / 20, 2) for x in range(10, 20)]
    if current is not None and current not in ths:
        ths = sorted({*ths, current})
    rows = []
    pos = sum(ys)
    for t in ths:
        passed = [p >= t for p in ps]
        tp = sum(1 for a, y in zip(passed, ys) if a and y)
        fp = sum(1 for a, y in zip(passed, ys) if a and not y)
        fn = sum(1 for a, y in zip(passed, ys) if not a and y)
        rows.append(
            ThresholdRow(
                threshold=t,
                auto_rate=sum(passed) / n,
                false_pass=fp / n,
                false_fail=fn / n,
                precision=tp / (tp + fp) if tp + fp else 0.0,
                recall=tp / pos if pos else 0.0,
            )
        )
    return Calibration(
        n=n, brier=brier, ece=ece, auroc=auroc, rows=rows, current_threshold=current, bins=bins
    )


def _auroc(ps: list[float], ys: list[bool]) -> float | None:
    pos = [p for p, y in zip(ps, ys) if y]
    neg = [p for p, y in zip(ps, ys) if not y]
    if not pos or not neg:
        return None
    # rank-based (Mann-Whitney U), handles ties
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    ranks = [0.0] * len(ps)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and ps[order[j + 1]] == ps[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    rank_sum = sum(ranks[i] for i, y in enumerate(ys) if y)
    u = rank_sum - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg))


async def acalibrate(
    rows: Sequence[dict[str, Any]],
    ev: Eval,
    label: str,
    backend: str | Backend | None = None,
    thresholds: Sequence[float] | None = None,
) -> Calibration:
    labels = [r.get(label) for r in rows]
    samples = [{k: v for k, v in r.items() if k != label} for r in rows]
    report = await aevaluate_dataset(samples, [ev], backend)
    current = getattr(ev, "threshold", None)
    return calibrate_from_report(
        report, ev.name, labels, thresholds, current if isinstance(current, float) else None
    )


def calibrate(
    rows: Sequence[dict[str, Any]], ev: Eval, label: str, backend: str | Backend | None = None, **kw: Any
) -> Calibration:
    return run_coro_sync(acalibrate(rows, ev, label, backend, **kw))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))
