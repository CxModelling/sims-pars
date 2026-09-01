"""A dedicated evaluator for the PCore expression sublanguage.

This is Phase 3's "written operator table and a dedicated evaluator" (see
``docs/spec/pcore.md``): it accepts the same AST subset as
:func:`sims_pars.util.safe_eval` but interprets it directly against an
explicit operator table instead of compiling and handing it to Python's
``eval()``. That makes the semantics a fixed, documented contract rather than
"whatever the running CPython does with a stripped-builtins eval" — and it
gives the lowering pipeline a way to compute a constant's value (for type
annotation checking, see ``lower.py``) without touching the runtime path
legacy scripts depend on for byte-identical results.

Operator precedence, highest to lowest (matches Python's, for the subset in
use here):

============  ================================  =================
precedence    operators                          associativity
============  ================================  =================
1 (highest)   function call, literal             --
2             ``**``                              right
3             unary ``+`` ``-`` ``not``            --
4             ``*`` ``/`` ``//`` ``%``             left
5             ``+`` ``-``                          left
6             ``==`` ``!=`` ``<`` ``<=`` ``>`` ``>=``  chained
7             ``and``                              left
8             ``or``                               left
9 (lowest)    ``a if c else b``                    --
============  ================================  =================
"""
from __future__ import annotations

import ast
import operator

from sims_pars.util import MATH_FUNC

__all__ = ['EvalError', 'evaluate', 'free_names']


class EvalError(ValueError):
    """Raised when the dedicated evaluator cannot compute an expression."""


# A tiny, side-effect-free subset of builtins -- kept in lockstep with
# sims_pars.util.safe_eval's _SAFE_BUILTINS.
_SAFE_BUILTINS = {
    'min': min, 'max': max, 'abs': abs, 'round': round,
    'sum': sum, 'pow': pow, 'len': len,
    'int': int, 'float': float, 'bool': bool,
}
_FUNCS = {**_SAFE_BUILTINS, **MATH_FUNC}

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg, ast.Not: operator.not_}
_CMP_OPS = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne,
    ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge,
}


def _ev(node: ast.AST, env: dict):
    if isinstance(node, ast.Expression):
        return _ev(node.body, env)

    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        if node.id.startswith('__'):
            raise EvalError(f"dunder identifier not allowed: {node.id!r}")
        try:
            return env[node.id]
        except KeyError:
            raise EvalError(f"undefined name: {node.id!r}") from None

    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise EvalError(f"unsupported unary operator: {type(node.op).__name__}")
        return op(_ev(node.operand, env))

    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise EvalError(f"unsupported operator: {type(node.op).__name__}")
        return op(_ev(node.left, env), _ev(node.right, env))

    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            result = True
            for v in node.values:
                result = _ev(v, env)
                if not result:
                    return result
            return result
        if isinstance(node.op, ast.Or):
            result = False
            for v in node.values:
                result = _ev(v, env)
                if result:
                    return result
            return result
        raise EvalError(f"unsupported boolean operator: {type(node.op).__name__}")

    if isinstance(node, ast.Compare):
        left = _ev(node.left, env)
        result = True
        for op_node, comparator in zip(node.ops, node.comparators):
            op = _CMP_OPS.get(type(op_node))
            if op is None:
                raise EvalError(f"unsupported comparison: {type(op_node).__name__}")
            right = _ev(comparator, env)
            result = op(left, right)
            if not result:
                return False
            left = right
        return result

    if isinstance(node, ast.IfExp):
        return _ev(node.body, env) if _ev(node.test, env) else _ev(node.orelse, env)

    if isinstance(node, ast.List):
        return [_ev(e, env) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_ev(e, env) for e in node.elts)
    if isinstance(node, ast.Set):
        return {_ev(e, env) for e in node.elts}
    if isinstance(node, ast.Dict):
        return {_ev(k, env): _ev(v, env) for k, v in zip(node.keys, node.values)}

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise EvalError('only direct function calls are allowed')
        name = node.func.id
        fn = _FUNCS.get(name, env.get(name))
        if fn is None or not callable(fn):
            raise EvalError(f"unknown function: {name!r}")
        args = [_ev(a, env) for a in node.args]
        kwargs = {kw.arg: _ev(kw.value, env) for kw in node.keywords}
        return fn(*args, **kwargs)

    raise EvalError(f"unsupported expression element: {type(node).__name__}")


def evaluate(expr: str, env: dict | None = None):
    """Evaluate ``expr`` against ``env`` without ever calling ``eval()``.

    Raises :class:`EvalError` for a disallowed construct or an undefined
    name (the latter is the expected outcome for anything but a closed
    constant expression -- callers use it to distinguish 'this is a
    compile-time constant' from 'this depends on parents').
    """
    try:
        tree = ast.parse(expr, mode='eval')
    except SyntaxError as e:
        raise EvalError(f"malformed expression: {e}") from e
    return _ev(tree, dict(env) if env else {})


def free_names(expr: str) -> set[str]:
    """The names ``expr`` references that are not a known function."""
    tree = ast.parse(expr, mode='eval')
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id not in _FUNCS}
