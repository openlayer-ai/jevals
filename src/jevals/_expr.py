"""A tiny safe expression evaluator for YAML policies and scores.

Supports: numbers, strings, booleans, names with dotted attribute access
(`action.approve`), comparisons, and/or/not, + - * /, min/max/abs/round,
and `x if c else y`. Nothing else. No calls except the whitelisted ones,
no attribute access on anything that isn't a plain value or mapping.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Mapping
from typing import Any

_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_CMP = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}
_FUNCS: dict[str, Any] = {
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "len": len,
    "float": float,
    "int": int,
    "bool": bool,
}


class ExprError(ValueError):
    pass


def _get(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        if name in obj:
            return obj[name]
        raise ExprError(f"unknown key {name!r}")
    if hasattr(obj, name) and not name.startswith("_"):
        v = getattr(obj, name)
        if callable(v) and not isinstance(v, (int, float, str, bool)):
            raise ExprError(f"{name!r} is not a value")
        return v
    raise ExprError(f"unknown name {name!r}")


class _Eval(ast.NodeVisitor):
    def __init__(self, env: Mapping[str, Any]):
        self.env = env

    def visit_Expression(self, n: ast.Expression) -> Any:
        return self.visit(n.body)

    def visit_Constant(self, n: ast.Constant) -> Any:
        if isinstance(n.value, (int, float, str, bool)) or n.value is None:
            return n.value
        raise ExprError("unsupported constant")

    def visit_Name(self, n: ast.Name) -> Any:
        if n.id in ("True", "False", "None"):
            return {"True": True, "False": False, "None": None}[n.id]
        return _get(self.env, n.id)

    def visit_Attribute(self, n: ast.Attribute) -> Any:
        return _get(self.visit(n.value), n.attr)

    def visit_Subscript(self, n: ast.Subscript) -> Any:
        base = self.visit(n.value)
        idx = self.visit(n.slice)
        try:
            return base[idx]
        except (KeyError, IndexError, TypeError) as e:
            raise ExprError(f"bad subscript {idx!r}") from e

    def visit_BinOp(self, n: ast.BinOp) -> Any:
        op = _BIN.get(type(n.op))
        if op is None:
            raise ExprError("unsupported operator")
        return op(self.visit(n.left), self.visit(n.right))

    def visit_UnaryOp(self, n: ast.UnaryOp) -> Any:
        v = self.visit(n.operand)
        if isinstance(n.op, ast.Not):
            return not v
        if isinstance(n.op, ast.USub):
            return -v
        if isinstance(n.op, ast.UAdd):
            return +v
        raise ExprError("unsupported unary operator")

    def visit_BoolOp(self, n: ast.BoolOp) -> Any:
        if isinstance(n.op, ast.And):
            for v in n.values:
                r = self.visit(v)
                if not r:
                    return r
            return r
        for v in n.values:
            r = self.visit(v)
            if r:
                return r
        return r

    def visit_Compare(self, n: ast.Compare) -> Any:
        left = self.visit(n.left)
        for op, right_node in zip(n.ops, n.comparators):
            right = self.visit(right_node)
            fn = _CMP.get(type(op))
            if fn is None:
                raise ExprError("unsupported comparison")
            if not fn(left, right):
                return False
            left = right
        return True

    def visit_IfExp(self, n: ast.IfExp) -> Any:
        return self.visit(n.body) if self.visit(n.test) else self.visit(n.orelse)

    def visit_Call(self, n: ast.Call) -> Any:
        if not isinstance(n.func, ast.Name) or n.func.id not in _FUNCS:
            raise ExprError("only min/max/abs/round/len/float/int/bool may be called")
        return _FUNCS[n.func.id](*[self.visit(a) for a in n.args])

    def visit_List(self, n: ast.List) -> Any:
        return [self.visit(e) for e in n.elts]

    def visit_Tuple(self, n: ast.Tuple) -> Any:
        return tuple(self.visit(e) for e in n.elts)

    def generic_visit(self, n: ast.AST) -> Any:
        raise ExprError(f"unsupported syntax: {type(n).__name__}")


def safe_eval(expr: str, env: Mapping[str, Any]) -> Any:
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise ExprError(f"bad expression {expr!r}: {e.msg}") from e
    return _Eval(env).visit(tree)


def validate_expr(expr: str) -> None:
    """Parse-only check, used by `jevals validate`."""
    tree = ast.parse(expr.strip(), mode="eval")
    for node in ast.walk(tree):
        if isinstance(
            node,
            (ast.Lambda, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp, ast.Await, ast.Yield),
        ):
            raise ExprError(f"unsupported syntax in {expr!r}: {type(node).__name__}")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in _FUNCS):
            raise ExprError(f"disallowed call in {expr!r}")
