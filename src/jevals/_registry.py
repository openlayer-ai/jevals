"""Every built-in eval by name, for the CLI and MCP server."""

from __future__ import annotations

import inspect
from typing import Any

from . import agent, quality, security
from ._eval import Eval


def builtin_evals() -> dict[str, type[Eval]]:
    out: dict[str, type[Eval]] = {}
    for mod in (agent, security, quality):
        for name in getattr(mod, "__all__", []):
            obj = getattr(mod, name)
            if inspect.isclass(obj) and issubclass(obj, Eval):
                out[obj.name or name] = obj
    return out


def get_eval(name: str, **kw: Any) -> Eval:
    """`get_eval("grounded")`, `get_eval("Grounded")`, `get_eval("security.PII", action="redact")`."""
    evals = builtin_evals()
    key = name.split(".")[-1]
    if key in evals:
        return evals[key](**kw)
    for cls in evals.values():
        if cls.__name__ == key:
            return cls(**kw)
    raise KeyError(f"no eval named {name!r}; try one of {sorted(evals)}")


def resolve_evals(specs: list[Any]) -> list[Eval]:
    """Mixed list of Eval instances, names, {"name": ..., **kwargs} dicts, or paths to YAML files."""
    from ._declarative import load_evals

    out: list[Eval] = []
    for spec in specs:
        if isinstance(spec, Eval):
            out.append(spec)
        elif isinstance(spec, str):
            if spec.endswith((".yaml", ".yml", ".json")) or "/" in spec:
                out.extend(load_evals(spec))
            else:
                out.append(get_eval(spec))
        elif isinstance(spec, dict):
            d = dict(spec)
            if "questions" in d:
                from ._declarative import DeclarativeEval

                out.append(DeclarativeEval(d))
            else:
                out.append(get_eval(d.pop("name"), **d))
        else:
            raise TypeError(f"cannot resolve eval from {spec!r}")
    return out
