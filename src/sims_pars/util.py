import ast
import math

import numpy as np
import scipy.special as sp
from numpy.random import choice

__author__ = 'TimeWz667'
__all__ = ['add_math_func', 'MATH_FUNC',
           'add_data_func', 'find_data_sampler', 'DATA_FUNC',
           'ScriptException', 'resample', 'safe_eval',
           'parse_parents', 'parse_math_expression',
           'parse_function', 'evaluate_function']


# --- restricted expression evaluation -----------------------------------------

# AST node types allowed inside a sims-pars script expression. Everything that
# could reach the host environment (attribute access, subscripting, lambdas,
# comprehensions, walrus, f-strings, ...) is rejected.
_ALLOWED_NODES = (
    ast.Expression, ast.Constant, ast.Name, ast.Load,
    ast.Call, ast.keyword,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
    ast.IfExp, ast.List, ast.Tuple, ast.Dict, ast.Set,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not,
    ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)


class UnsafeExpression(ValueError):
    """Raised when a script expression contains a disallowed construct."""


# A tiny, side-effect-free subset of builtins that script expressions may use.
_SAFE_BUILTINS = {
    'min': min, 'max': max, 'abs': abs, 'round': round,
    'sum': sum, 'pow': pow, 'len': len,
    'int': int, 'float': float, 'bool': bool,
}


def _validate_expr(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise UnsafeExpression(
                f'Disallowed expression element: {type(node).__name__}'
            )
        if isinstance(node, ast.Name) and node.id.startswith('__'):
            raise UnsafeExpression(f'Dunder identifier not allowed: {node.id}')
        if isinstance(node, ast.Call) and not isinstance(node.func, ast.Name):
            raise UnsafeExpression('Only direct function calls are allowed')


def safe_eval(expr, names=None, local=None):
    """
    Evaluate a sims-pars script expression with builtins stripped and the AST
    restricted to an arithmetic / function-call allow-list.

    :param expr: expression string or a pre-compiled code object
    :param names: mapping of names available to the expression (functions,
                  distribution creators, ...)
    :param local: mapping of local values (parent nodes)
    :return: the evaluated result
    """
    if isinstance(expr, str):
        tree = ast.parse(expr, mode='eval')
        _validate_expr(tree)
        expr = compile(tree, '<sims-pars-expr>', 'eval')
    glb = {'__builtins__': dict(_SAFE_BUILTINS)}
    if names:
        glb.update(names)
    return eval(expr, glb, dict(local) if local else {})


def ifelse(cond, a, b):
    return a if cond else b


def step(key, cut, a, b):
    return a if key < cut else b


MATH_FUNC = {
    'hypot': np.hypot,
    'exp': np.exp,
    'log': np.log,
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan,
    'ceil': np.ceil,
    'floor': np.floor,
    'sqrt': np.sqrt,
    'abs': np.abs,
    'erf': math.erf,
    'pow': math.pow,
    'logit': sp.logit,
    'expit': sp.expit,
    'ifelse': ifelse,
    'step': step
}


def add_math_func(fn_name, fn):
    """
    Add a customised data function
    :param fn_name: name of the function
    :param fn: a callable function
    """
    if callable(fn):
        MATH_FUNC[fn_name] = fn


DATA_FUNC = dict()


def add_data_func(fn_name, fn):
    assert callable(fn) or callable(fn.get_sampler)
    assert fn_name not in DATA_FUNC
    DATA_FUNC[fn_name] = fn


def find_data_sampler(fn_name, loc):
    fn = DATA_FUNC[fn_name]
    return fn.get_sampler(loc)


class ScriptException(Exception):
    def __init__(self, err):
        self.Err = err

    def __repr__(self):
        return self.Err


def resample(wts, hs, pars=None, log=True, new_size=None):
    size = len(wts)
    new_size = max(new_size, 1) if new_size else size

    fin = [np.isfinite(wt) for wt in wts]
    wts = [float(wt) for wt, f in zip(wts, fin) if f]
    size = len(wts)
    if size == 0:
        if pars:
            return hs, pars, -np.log(size)
        else:
            return hs, -np.log(size)

    hs = [h for h, f in zip(hs, fin) if f]

    if pars:
        pars = [par for par, f in zip(pars, fin) if f]

    if log:
        wts = np.array(wts)
        lse = sp.logsumexp(wts)
        wts -= lse
        sel = choice(size, new_size, replace=True, p=np.exp(wts))
    else:
        lse = np.sum(wts)
        wts /= lse
        lse = np.log(lse)
        sel = choice(size, new_size, replace=True, p=wts)
    if pars:
        return [hs[i] for i in sel], [pars[i] for i in sel], lse - np.log(new_size)
    else:
        return [hs[i] for i in sel], lse - np.log(new_size)


def find_ast_parents(seq_ast):
    v, f = set(), set()

    for s in ast.walk(seq_ast):
        if isinstance(s, ast.Name):
            v.add(s.id)
        elif isinstance(s, ast.Call):
            f.add(s.func.id)
    return v - f, f


def parse_parents(seq):
    return find_ast_parents(ast.parse(seq))


class MathExpression:
    def __init__(self, eq, var, fn):
        self.Expression = eq
        self.Var = var
        self.Func = fn
        tree = ast.parse(eq, mode='eval')
        _validate_expr(tree)
        self._code = compile(tree, '<sims-pars-expr>', 'eval')

    def __call__(self, loc=None):
        try:
            return self.execute(loc)
        except NameError:
            return self.Expression

    def execute(self, loc=None):
        return safe_eval(self._code, MATH_FUNC, loc)

    @property
    def Parents(self):
        return self.Var

    def is_executable(self, loc):
        return all(v in loc for v in self.Var) and all(f in MATH_FUNC for f in self.Func)

    def __str__(self):
        return self.Expression

    __repr__ = __str__


def parse_math_expression(seq):
    v, f = parse_parents(seq)
    return MathExpression(seq, v, f)


def ast_to_math_expression(seq_ast, seq=None):
    v, f = find_ast_parents(seq_ast)
    seq = seq if seq else ast.unparse(seq_ast)
    return MathExpression(seq, v, f)


class ParsedFunction:
    def __init__(self, src, fn, args):
        self.Source = src
        self.Function = fn
        self.Arguments = args

    def get_arguments(self, loc=None):
        args = list()
        for arg in self.Arguments:
            # todo opti
            arg = dict(arg)
            try:
                arg['value'] = arg['value'].execute(loc)
            except NameError:
                raise NameError("Parent nodes are not fully defined")
            args.append(arg)
        return args

    def to_blueprint(self, name, loc=None):
        return {
            'Name': name,
            'Type': self.Function,
            'Args': self.get_arguments(loc)
        }

    @property
    def Parents(self):
        vs = [arg['value'].Var for arg in self.Arguments]
        return set.union(*vs) if vs else set()

    def to_json(self, loc=None):
        return {
            'Source': self.Source,
            'Type': self.Function,
            'Args': [arg['value'](loc) for arg in self.Arguments]
        }

    def __str__(self):
        return self.Source

    __repr__ = __str__


def _compact(seq: str) -> str:
    """Strip whitespace, but never inside a string literal."""
    out, quote, i = [], "", 0
    while i < len(seq):
        c = seq[i]
        if quote:
            out.append(c)
            if c == "\\" and i + 1 < len(seq):
                out.append(seq[i + 1])
                i += 2
                continue
            if c == quote:
                quote = ""
        elif c in "\"'":
            quote = c
            out.append(c)
        elif not c.isspace():
            out.append(c)
        i += 1
    return "".join(out)


def parse_function(seq):
    seq = _compact(seq)
    body = ast.parse(seq, mode='eval').body  # propagates SyntaxError

    if isinstance(body, ast.Call) and isinstance(body.func, ast.Name):
        f = body.func.id
        pars = [{'value': ast_to_math_expression(a)} for a in body.args]
        pars += [
            {'key': kw.arg, 'value': ast_to_math_expression(kw.value)}
            for kw in body.keywords
        ]
    else:
        # Not a direct call (e.g. a bare expression handed to PseudoLoci).
        # Keep it tolerant: expose the free names as the "arguments".
        names, _ = find_ast_parents(body)
        f = None
        pars = [
            {'value': ast_to_math_expression(ast.Name(id=n, ctx=ast.Load()))}
            for n in sorted(names)
        ]
    return ParsedFunction(seq, f, pars)


class EvaluatedFunction:
    def __init__(self, src, fn, args):
        self.Source = src
        self.Function = fn
        self.Arguments = args

    def to_blueprint(self, name):
        return {
            'Name': name,
            'Type': self.Function,
            'Args': self.Arguments
        }

    def to_json(self):
        return {
            'Source': self.Source,
            'Type': self.Function,
            'Args': [arg['value'] for arg in self.Arguments]
        }

    def __str__(self):
        return self.Source

    __repr__ = __str__


def evaluate_function(pf: ParsedFunction, loc=None):
    args = pf.get_arguments(loc)
    return EvaluatedFunction(pf.Source, pf.Function, args)


if __name__ == '__main__':

    print('Find parents')
    print(parse_parents('x+y/2 * max(z, 5)'), '\n')

    print('Math expression')
    me = parse_math_expression('x+y/2 * max(z, 5)')
    print(me)
    print(me({'x': 2, 'y': 4, 'z': 10}), '\n')

    func = parse_function('my_func(4*a, "k", k, t=5, s=False)')
    print(func)
    print(func.to_json())
    print(func.to_json({'k': 7}))
    print(func.to_json({'k': 7, 'a': 10}))

    func = evaluate_function(func, {'k': 7, 'a': 10})
    print(func.to_json(), '\n')

    print(resample([0, -1.2, -1], ['a', 'b', 'c']))

    func = parse_math_expression('ifelse(y > 10, 0, 100)')
    print(func({'y': 5}))
    print(func({'y': 15}))
