"""MCP server so coding agents (Cursor, Claude Code, Copilot) can run and write evals.

    jevals mcp                # stdio server
    jevals mcp --install      # write the entry into .cursor/mcp.json / Claude config

Tools: list_evals, describe_eval, evaluate, evaluate_file, gate, author_eval, validate_eval, schema, docs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .._declarative import DeclarativeEval, eval_schema
from .._gate import Gate
from .._registry import builtin_evals, resolve_evals
from .._runner import aevaluate, aevaluate_dataset
from ._docs import llm_docs


def build_server() -> Any:
    try:
        from mcp.server.mcpserver import MCPServer as FastMCP  # type: ignore  # mcp 2.x
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP  # type: ignore  # mcp 1.x
        except ImportError as e:
            raise SystemExit('mcp is not installed: pip install "jevals[mcp]"') from e

    mcp = FastMCP(
        "jevals",
        instructions="Run and author evals and guardrails for AI agents. Call `docs` first if you have not used jevals before.",
    )

    @mcp.tool()
    def list_evals() -> list[dict[str, Any]]:
        """Every built-in eval: name, category, one-line description, required sample fields."""
        # Read the class, like `jevals list`: CustomRubric can't be built without arguments.
        return [
            {
                "name": name,
                "category": cls.category,
                "description": cls.description or (cls.__doc__ or "").strip().split("\n")[0],
                "requires": list(cls.requires),
            }
            for name, cls in builtin_evals().items()
        ]

    @mcp.tool()
    def describe_eval(name: str) -> dict[str, Any]:
        """Details for one eval, including the questions it asks (for a sample with generic content)."""
        from .._sample import Sample

        ev = resolve_evals([name])[0]
        d = ev.describe()
        try:
            s = Sample(
                messages=[
                    {"role": "user", "content": "<user request>"},
                    {"role": "assistant", "content": "<final answer>"},
                ],
                tool_call={"name": "<tool>", "args": {}},
                contexts=["<context>"],
                expected="<expected>",
            )
            d["state_keys"] = sorted(ev.state(s).keys())
            d["questions"] = {k: q.model_dump() for k, q in ev.questions(s).items()}
        except Exception as e:  # noqa: BLE001
            d["note"] = f"questions depend on the sample: {e}"
        return d

    @mcp.tool()
    async def evaluate(
        sample: dict[str, Any], evals: list[Any], backend: str | None = None
    ) -> dict[str, Any]:
        """Run evals on one sample. `sample` has `messages` (OpenAI chat format) and optionally `tools`,
        `tool_call`, `contexts`, `expected`. `evals` is a list of names ("agent.grounded"), YAML paths,
        or inline eval specs (dicts with `questions`)."""
        rep = await aevaluate(sample, resolve_evals(evals), backend)
        return {**rep.to_dict(), "table": rep.table()}

    @mcp.tool()
    async def evaluate_file(
        path: str,
        evals: list[Any],
        backend: str | None = None,
        limit: int | None = None,
        out: str | None = None,
    ) -> dict[str, Any]:
        """Run evals over a JSONL file of samples. Returns the summary table; writes per-row results to `out` if given."""
        from .._runner import load_jsonl

        rows = load_jsonl(path)
        if limit:
            rows = rows[:limit]
        rep = await aevaluate_dataset(rows, resolve_evals(evals), backend)
        if out:
            rep.write_jsonl(out)
        return {"summary": rep.summary(), "table": rep.table(), "n": rep.n}

    @mcp.tool()
    async def gate(sample: dict[str, Any], evals: list[Any], backend: str | None = None) -> dict[str, Any]:
        """Run evals as a gate and return {action: allow|block|escalate|modify, reasons, value}."""
        d = await Gate(*resolve_evals(evals), backend=backend).acheck(sample)
        return d.model_dump(exclude={"results"}) | {
            "results": {r.name: r.model_dump(exclude={"answers", "usage"}) for r in d.results}
        }

    @mcp.tool()
    def validate_eval(spec: dict[str, Any] | str) -> dict[str, Any]:
        """Validate a YAML/JSON eval spec (dict, YAML text, or a file path). Returns {ok, errors, eval}."""
        import yaml

        try:
            if isinstance(spec, str):
                if os.path.exists(spec):
                    spec = yaml.safe_load(Path(spec).read_text())
                else:
                    spec = yaml.safe_load(spec)
            ev = DeclarativeEval(spec)  # type: ignore[arg-type]
            return {"ok": True, "errors": [], "eval": ev.describe()}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "errors": [f"{type(e).__name__}: {e}"]}

    @mcp.tool()
    def author_eval(spec: dict[str, Any], path: str, overwrite: bool = False) -> dict[str, Any]:
        """Validate an eval spec and write it to `path` as YAML. Use `schema` for the format.
        For contract evals: one rule per question; cite it with covers: [r-…] (ids from
        contract_rules); write the question so true means the agent did the named thing, and
        check an absence as a question that is true when the violation happened, never "No …"."""
        import yaml

        ev = DeclarativeEval(spec)
        p = Path(path)
        if p.exists() and not overwrite:
            return {"ok": False, "errors": [f"{p} exists; pass overwrite=true"]}
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True, width=120))
        return {"ok": True, "path": str(p), "eval": ev.describe()}

    @mcp.tool()
    async def contract_rules(
        prompt: str | None = None,
        path: str | None = None,
        trace: str | None = None,
        out: str | None = None,
        use_model: bool = True,
        backend: str | None = None,
    ) -> dict[str, Any]:
        """Split an agent's system prompt into rules with stable ids (r-3f2a9c). Give the prompt
        text, a file `path`, or a JSONL `trace`. Write evals whose questions cite the ids in
        `covers`, one rule per question, then run contract_check."""
        from ..contract import aextract_rules, prompt_from_trace

        if trace:
            text, source = prompt_from_trace(trace), f"trace:{trace}#0"
        elif path:
            text, source = Path(path).read_text(), path
        elif prompt:
            text, source = prompt, None
        else:
            return {"ok": False, "errors": ["give prompt, path or trace"]}
        ex = await aextract_rules(text, backend=backend, use_model=use_model, source=source)
        if out:
            ex.contract.dump(out)
        return {
            "ok": True,
            "rules": [r.model_dump() for r in ex.contract.rules],
            "dropped": ex.dropped,
            "written": out,
            "usage": str(ex.usage),
        }

    @mcp.tool()
    async def contract_check(
        contract: str,
        evals: list[str],
        prompt: str | None = None,
        use_model: bool = True,
        strict: bool = False,
        backend: str | None = None,
    ) -> dict[str, Any]:
        """Check evals against contract.yaml: uncovered rules, stale citations, rules edited under
        their old id, changed prompt
        (pass the current prompt text), and with use_model, citations a decision model doubts."""
        from .._declarative import load_evals
        from ..contract import acheck_coverage, load_contract

        loaded = [e for p in evals for e in load_evals(p)]
        rep = await acheck_coverage(
            load_contract(contract), loaded, backend=backend, use_model=use_model, prompt=prompt
        )
        return {
            "ok": not rep.failed(strict),
            "failed": rep.failed(strict),
            "report": rep.model_dump(mode="json"),
        }

    @mcp.tool()
    def schema() -> dict[str, Any]:
        """JSON Schema for a YAML eval file."""
        return eval_schema()

    @mcp.tool()
    def docs() -> str:
        """Compact jevals reference: API, sample format, eval list, backends, how to write an eval."""
        return llm_docs()

    return mcp


def serve() -> None:
    build_server().run()


def install(target: str | None = None, cwd: str | Path | None = None) -> list[Path]:
    """Write the server entry into MCP configs. target: cursor | claude | vscode | all (default: whatever exists, else cursor)."""
    cwd = Path(cwd or os.getcwd())
    entry = {"command": "jevals", "args": ["mcp"]}
    targets: dict[str, tuple[Path, str]] = {
        "cursor": (cwd / ".cursor" / "mcp.json", "mcpServers"),
        "claude": (cwd / ".mcp.json", "mcpServers"),
        "vscode": (cwd / ".vscode" / "mcp.json", "servers"),
    }
    chosen = (
        list(targets)
        if target == "all"
        else [target]
        if target
        else [k for k, (p, _) in targets.items() if p.exists()] or ["cursor"]
    )
    written = []
    for k in chosen:
        path, key = targets[k]
        cfg: dict[str, Any] = {}
        if path.exists():
            try:
                cfg = json.loads(path.read_text() or "{}")
            except json.JSONDecodeError:
                cfg = {}
        cfg.setdefault(key, {})["jevals"] = entry
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cfg, indent=2) + "\n")
        written.append(path)
    return written
