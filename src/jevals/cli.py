"""jevals command line.

jevals run traces.jsonl --evals agent.tool_choice,agent.grounded,security.phi,evals/mine.yaml
jevals check sample.json --evals ...            one sample, full table
jevals gate sample.json --evals ...             one sample, decision
jevals list [--category agent]
jevals describe grounded
jevals validate evals/*.yaml
jevals schema
jevals docs --llm
jevals calibrate labeled.jsonl --eval evals/tool_call_risk.yaml --label human_decision
jevals mcp [--install [cursor|claude|vscode|all]]
jevals hook pre|post --evals ...                Claude Code hook (stdin JSON -> stdout JSON)
jevals bench [--ragas] [--n 50]                 replay a fixed dataset, print measured numbers
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _evals(spec: str | None) -> list[Any]:
    from ._registry import resolve_evals

    if not spec:
        raise SystemExit("--evals is required (comma-separated names or YAML paths)")
    return resolve_evals([s.strip() for s in spec.split(",") if s.strip()])


def _load_sample(path: str) -> dict[str, Any]:
    text = sys.stdin.read() if path == "-" else Path(path).read_text()
    return json.loads(text)


def cmd_run(a: argparse.Namespace) -> int:
    from ._runner import evaluate_dataset, load_jsonl

    rows = load_jsonl(a.path)
    if a.limit:
        rows = rows[: a.limit]
    evals = _evals(a.evals)
    n_done = [0]

    def progress(i: int, rep: Any) -> None:
        n_done[0] += 1
        if not a.quiet and (n_done[0] % 25 == 0 or n_done[0] == len(rows)):
            print(f"\r{n_done[0]}/{len(rows)}", end="", file=sys.stderr, flush=True)

    rep = evaluate_dataset(rows, evals, a.backend, concurrency=a.concurrency, on_result=progress)
    if not a.quiet:
        print(file=sys.stderr)
    if a.out:
        rep.write_jsonl(a.out)
    if a.json:
        print(
            json.dumps(
                {"summary": rep.summary(), "usage": rep.usage.model_dump(), "wall_ms": rep.wall_ms},
                indent=2,
                default=str,
            )
        )
    else:
        print(rep.table())
        if a.show_failures:
            for name in {r.name for r in rep.reports[0].results} if rep.reports else set():
                fails = rep.failures(name)
                if fails:
                    print(f"\n{name}: {len(fails)} failures")
                    for i, r in fails[: a.show_failures]:
                        print(f"  [{i}] {r.detail or r.answer or r.score}")
    return 0


def cmd_check(a: argparse.Namespace) -> int:
    from ._runner import evaluate

    rep = evaluate(_load_sample(a.path), _evals(a.evals), a.backend)
    print(json.dumps(rep.to_dict(), indent=2, default=str) if a.json else rep.table())
    return 0 if rep.passed else 1


def cmd_gate(a: argparse.Namespace) -> int:
    from ._gate import Gate

    d = Gate(*_evals(a.evals), backend=a.backend).check(_load_sample(a.path))
    if a.json:
        print(json.dumps(d.model_dump(exclude={"results"}), indent=2, default=str))
    else:
        print(d)
        for r in d.results:
            print(" ", r)
    return {"allow": 0, "modify": 0, "escalate": 2, "block": 3}[d.action]


def cmd_list(a: argparse.Namespace) -> int:
    from ._registry import builtin_evals

    rows = [
        (cls.category, name, (cls.__doc__ or "").strip().split("\n")[0])
        for name, cls in builtin_evals().items()
    ]
    if a.category:
        rows = [r for r in rows if r[0] == a.category]
    if a.json:
        print(json.dumps([{"category": c, "name": n, "description": d} for c, n, d in rows], indent=2))
        return 0
    for cat in sorted({r[0] for r in rows}):
        print(cat)
        for c, n, d in sorted(rows):
            if c == cat:
                print(f"  {n:<24} {d}")
    return 0


def cmd_describe(a: argparse.Namespace) -> int:
    from ._registry import resolve_evals
    from ._sample import Sample

    ev = resolve_evals([a.name])[0]
    d = ev.describe()
    s = Sample(
        messages=[
            {"role": "system", "content": "<system prompt>"},
            {"role": "user", "content": "<user request>"},
            {"role": "assistant", "content": "<final answer>"},
        ],
        tool_call={"name": "<tool>", "args": {"x": 1}},
        contexts=["<context>"],
        expected="<expected>",
        expected_tool_calls=[],
    )
    try:
        d["state_keys"] = sorted(ev.state(s).keys())
        d["questions"] = {k: q.model_dump() for k, q in ev.questions(s).items()}
    except Exception as e:  # noqa: BLE001
        d["note"] = f"questions depend on the sample: {e}"
    print(json.dumps(d, indent=2, default=str))
    return 0


def cmd_validate(a: argparse.Namespace) -> int:
    from ._declarative import load_eval

    bad = 0
    for p in a.paths:
        try:
            ev = load_eval(p)
            print(f"ok    {p}  ({ev.name}, {len(ev._questions)} questions)")
        except Exception as e:  # noqa: BLE001
            bad += 1
            print(f"FAIL  {p}  {type(e).__name__}: {e}")
    return 1 if bad else 0


def cmd_schema(a: argparse.Namespace) -> int:
    from ._declarative import eval_schema

    print(json.dumps(eval_schema(), indent=2))
    return 0


def cmd_docs(a: argparse.Namespace) -> int:
    from .integrations._docs import llm_docs

    print(llm_docs())
    return 0


def cmd_calibrate(a: argparse.Namespace) -> int:
    from ._calibrate import calibrate
    from ._runner import load_jsonl

    rows = load_jsonl(a.path)
    if a.limit:
        rows = rows[: a.limit]
    ev = _evals(a.eval)[0]
    cal = calibrate(rows, ev, a.label, a.backend)
    if a.json:
        print(cal.model_dump_json(indent=2))
    else:
        print(cal.table())
        best = cal.best(a.max_false_pass)
        if best:
            print(
                f"\nHighest auto-pass with wrong passes <= {a.max_false_pass * 100:.1f}%: threshold {best.threshold:.2f}"
            )
    return 0


def cmd_mcp(a: argparse.Namespace) -> int:
    from .integrations import mcp_server

    if a.install is not None:
        for p in mcp_server.install(a.install or None):
            print(f"wrote {p}")
        return 0
    mcp_server.serve()
    return 0


def cmd_hook(a: argparse.Namespace) -> int:
    from ._gate import Gate
    from .integrations.claude_agent import run_hook_command

    return run_hook_command(Gate(*_evals(a.evals), backend=a.backend), a.phase)


def cmd_bench(a: argparse.Namespace) -> int:
    from ._bench import run_bench

    return run_bench(n=a.n, backend=a.backend, ragas=a.ragas, model=a.model)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="jevals", description="Agent evals and guardrails in one request.")
    p.add_argument("--backend", help="jev | vercel | kev://host:port | laya | llm:<model> | mock")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="evaluate a JSONL dataset")
    r.add_argument("path")
    r.add_argument("--evals", required=True)
    r.add_argument("--out")
    r.add_argument("--limit", type=int)
    r.add_argument("--concurrency", type=int, default=16)
    r.add_argument("--show-failures", type=int, default=0, metavar="N")
    r.add_argument("--json", action="store_true")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(fn=cmd_run)

    c = sub.add_parser("check", help="evaluate one sample (JSON file or -)")
    c.add_argument("path")
    c.add_argument("--evals", required=True)
    c.add_argument("--json", action="store_true")
    c.set_defaults(fn=cmd_check)

    g = sub.add_parser("gate", help="gate one sample; exit 0 allow, 2 escalate, 3 block")
    g.add_argument("path")
    g.add_argument("--evals", required=True)
    g.add_argument("--json", action="store_true")
    g.set_defaults(fn=cmd_gate)

    li = sub.add_parser("list", help="list built-in evals")
    li.add_argument("--category")
    li.add_argument("--json", action="store_true")
    li.set_defaults(fn=cmd_list)

    d = sub.add_parser("describe", help="show an eval's state keys and questions")
    d.add_argument("name")
    d.set_defaults(fn=cmd_describe)

    v = sub.add_parser("validate", help="validate YAML/JSON eval files")
    v.add_argument("paths", nargs="+")
    v.set_defaults(fn=cmd_validate)

    sub.add_parser("schema", help="JSON Schema for YAML evals").set_defaults(fn=cmd_schema)

    dd = sub.add_parser("docs", help="print a compact reference")
    dd.add_argument("--llm", action="store_true", help="(default) format for pasting into an agent's context")
    dd.set_defaults(fn=cmd_docs)

    ca = sub.add_parser("calibrate", help="threshold table and calibration metrics against labels")
    ca.add_argument("path")
    ca.add_argument("--eval", required=True)
    ca.add_argument("--label", required=True)
    ca.add_argument("--limit", type=int)
    ca.add_argument("--max-false-pass", type=float, default=0.01)
    ca.add_argument("--json", action="store_true")
    ca.set_defaults(fn=cmd_calibrate)

    m = sub.add_parser("mcp", help="MCP server for coding agents")
    m.add_argument("--install", nargs="?", const="", metavar="cursor|claude|vscode|all")
    m.set_defaults(fn=cmd_mcp)

    h = sub.add_parser("hook", help="Claude Code hook: stdin event -> stdout decision")
    h.add_argument("phase", choices=["pre", "post"])
    h.add_argument("--evals", required=True)
    h.set_defaults(fn=cmd_hook)

    b = sub.add_parser("bench", help="measure requests, tokens, latency on a fixed dataset")
    b.add_argument("--n", type=int, default=20)
    b.add_argument(
        "--ragas",
        action="store_true",
        help="also run Ragas on the same rows (pip install ragas datasets langchain-openai 'langchain-community<0.4'; "
        "OPENAI_API_KEY or OPENROUTER_API_KEY)",
    )
    b.add_argument("--model", default="gpt-4.1-mini")
    b.set_defaults(fn=cmd_bench)

    a = p.parse_args(argv)
    try:
        return a.fn(a)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
