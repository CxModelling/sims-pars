"""Lowering: a parsed :class:`Program` -> :class:`BayesianNetwork`.

Phase 1 reuses the legacy loci classes and ``BayesianNetwork.append_loci``, so a
valid script produces a byte-identical network. The value it adds is that every
failure that used to be a raw exception (unknown distribution tag, malformed
expression, a cycle) becomes a diagnostic with a source span.

Phase 3 adds ``: type`` annotation checking and Phase 4 adds plate expansion
(handled in ``parser.py``, before this module ever sees the statements) and
``include`` composition (handled here, since it needs the filesystem).
"""
from __future__ import annotations

import difflib
import os
import re

from sims_pars.pcore.ast import IncludeDef, Model, NodeDef
from sims_pars.pcore.diagnostics import DiagnosticError, Diagnostics
from sims_pars.pcore.parser import Program, parse

__all__ = ['to_network', 'compile_script', 'check']

_LEADING_IDENT = re.compile(r"[A-Za-z_]\w*")

# Phase 3: type annotations -----------------------------------------------
_KNOWN_TYPES = {'float', 'int', 'bool', 'vector', 'simplex'}


def _type_matches(tp: str, value) -> bool:
    if tp == 'float':
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if tp == 'int':
        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            return True
        return isinstance(value, float) and value.is_integer()
    if tp == 'bool':
        return isinstance(value, bool)
    if tp == 'vector':
        return isinstance(value, (list, tuple)) and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
    if tp == 'simplex':
        return (_type_matches('vector', value) and len(value) > 0
                and all(v >= 0 for v in value) and abs(sum(value) - 1) < 1e-6)
    return True  # unreachable once the E0300 check below has run


def _check_annotation(stmt: NodeDef, diags: Diagnostics) -> None:
    """Validate a node's ``: type`` annotation (Phase 3).

    Only a constant '=' right-hand side can be checked statically; a
    distribution ('~'), a function of parents, or an exogenous node has no
    value until sample time, so the annotation is recorded but not verified
    against a value.
    """
    if stmt.type_ann is None:
        return
    if stmt.type_ann not in _KNOWN_TYPES:
        near = difflib.get_close_matches(stmt.type_ann, sorted(_KNOWN_TYPES), n=1)
        hint = f"did you mean {near[0]!r}?" if near else \
            f"known types: {', '.join(sorted(_KNOWN_TYPES))}"
        diags.error(stmt.type_span or stmt.span, "E0300",
                    f"unknown type annotation {stmt.type_ann!r}", hint=hint)
        return
    if not stmt.is_assign or not stmt.rhs:
        return

    from sims_pars.pcore.evaluator import EvalError, evaluate
    try:
        value = evaluate(_strip(stmt.rhs))
    except EvalError:
        return  # depends on parents -- nothing to check until sample time
    except Exception:  # noqa: BLE001 - be permissive; build() reports real errors
        return

    if not _type_matches(stmt.type_ann, value):
        diags.warning(stmt.rhs_span or stmt.span, "E0301",
                      f"{stmt.name!r} is declared ': {stmt.type_ann}' but evaluates to {value!r}")


def _strip(text: str) -> str:
    """Remove insignificant whitespace, but never inside a string literal.

    The legacy parser did ``line.replace(' ', '')`` unconditionally, which
    corrupted ``cat({"low risk": 1})``. This keeps quoted contents verbatim and
    is otherwise identical for scripts with no strings (i.e. the whole corpus)."""
    out: list[str] = []
    quote = ""
    i = 0
    while i < len(text):
        c = text[i]
        if quote:
            out.append(c)
            if c == "\\" and i + 1 < len(text):
                out.append(text[i + 1])
                i += 2
                continue
            if c == quote:
                quote = ""
            i += 1
            continue
        if c in "\"'":
            quote = c
            out.append(c)
        elif not c.isspace():
            out.append(c)
        i += 1
    return "".join(out)


def _dist_tags() -> list[str]:
    from sims_pars.prob import DistributionCentre
    return DistributionCentre.list()


# Phase 4: composition ------------------------------------------------------

def _resolve_include(inc: IncludeDef, diags: Diagnostics, base_dir: str | None, seen: set):
    base_dir = base_dir or os.getcwd()
    full = inc.path if os.path.isabs(inc.path) else os.path.join(base_dir, inc.path)
    full = os.path.normpath(full)

    if full in seen:
        diags.error(inc.span, "E0251", f"circular include of {inc.path!r}")
        return None
    if not os.path.isfile(full):
        diags.error(inc.span, "E0250", f"included file not found: {inc.path!r}",
                    hint=f"looked for {full}")
        return None
    try:
        with open(full, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        diags.error(inc.span, "E0252", f"could not read {inc.path!r}: {e}")
        return None

    sub_program = parse(text)
    sub_model = sub_program.model
    if sub_model is None:
        diags.error(inc.span, "E0253", f"{inc.path!r} has no 'PCore' block to include")
        return None

    sub_diags = Diagnostics(list(sub_program.diagnostics))
    sub_bn = _lower_model(sub_model, sub_diags, base_dir=os.path.dirname(full),
                          seen=seen | {full})
    if any(d.is_error for d in sub_diags.sorted):
        first = next(d for d in sub_diags.sorted if d.is_error)
        diags.error(inc.span, "E0254", f"{inc.path!r} has errors: {first.message}")
        return None
    return sub_bn


def _splice_include(bn, inc_bn) -> None:
    """Merge every non-exogenous node of ``inc_bn`` into ``bn`` (same shape as
    :meth:`BayesianNetwork.merge`, minus the copy/rename it does for two
    already-built networks -- here ``bn`` is still being assembled)."""
    for node in inc_bn.Order:
        if inc_bn.is_exogenous(node):
            continue
        if node in bn:
            bn.DAG.remove_node(node)
        bn.append_from_js(inc_bn[node].to_json())
    bn.UserDefinedFunctions.update(inc_bn.UserDefinedFunctions)


def _lower_model(model: Model, diags: Diagnostics, base_dir: str | None = None,
                  seen: set | None = None):
    from sims_pars.bayesnet import BayesianNetwork
    from sims_pars.bayesnet.loci import (
        DistributionLoci,
        ExoValueLoci,
        FunctionLoci,
        PseudoLoci,
        ValueLoci,
    )

    seen = seen if seen is not None else set()
    bn = BayesianNetwork(model.name)
    for inc in model.includes:
        inc_bn = _resolve_include(inc, diags, base_dir, seen)
        if inc_bn is not None:
            _splice_include(bn, inc_bn)

    tags = set(_dist_tags())

    def build(stmt: NodeDef):
        span = stmt.rhs_span or stmt.span
        if stmt.is_exogenous:
            return ExoValueLoci(stmt.name)

        rhs = _strip(stmt.rhs or "")
        if not rhs:
            diags.error(stmt.span, "E0201", "empty right-hand side")
            return None

        if stmt.is_sample:
            m = _LEADING_IDENT.match(rhs)
            tag = m.group(0) if m else None
            if tag is None or (m and rhs[m.end():m.end() + 1] != "("):
                diags.error(span, "E0210",
                            "a '~' definition must be 'name ~ dist(...)'")
                return None
            if tag not in tags:
                near = difflib.get_close_matches(tag, sorted(tags), n=1)
                hint = f"did you mean {near[0]!r}?" if near else \
                    f"known: {', '.join(sorted(tags))}"
                diags.error(span, "E0211", f"unknown distribution {tag!r}", hint=hint)
                return None
            try:
                return DistributionLoci(stmt.name, rhs)
            except KeyError as e:
                diags.error(span, "E0212",
                            f"missing or unknown argument for {tag!r}: {e}")
            except Exception as e:  # noqa: BLE001 - surface as a diagnostic
                diags.error(span, "E0213",
                            f"cannot build distribution {tag!r}: {e}")
            return None

        # deterministic ('=')
        if rhs.startswith("f("):
            try:
                return PseudoLoci(stmt.name, rhs)
            except Exception as e:  # noqa: BLE001
                diags.error(span, "E0220", f"invalid pseudo node: {e}")
                return None
        try:
            return ValueLoci(stmt.name, rhs)
        except NameError:
            pass
        except SyntaxError:
            diags.error(span, "E0221", "malformed expression")
            return None
        except Exception as e:  # noqa: BLE001 - e.g. disallowed construct
            diags.error(span, "E0222", f"invalid expression: {e}")
            return None
        try:
            return FunctionLoci(stmt.name, rhs)
        except Exception as e:  # noqa: BLE001
            diags.error(span, "E0223", f"invalid function: {e}")
            return None

    for stmt in model.statements:
        _check_annotation(stmt, diags)
        loci = build(stmt)
        if loci is None:
            continue
        try:
            if stmt.description:
                bn.append_loci(loci, Des=stmt.description)
            else:
                bn.append_loci(loci)
        except AttributeError:
            diags.error(stmt.span, "E0230",
                        f"node {stmt.name!r} creates a cyclic dependency")

    try:
        bn.complete()
    except Exception as e:  # noqa: BLE001 - defensive
        diags.error(model.span, "E0240", f"could not finalise the model: {e}")
    return bn


def to_network(program: Program, strict: bool = True, path: str | None = None):
    """Lower a parsed program to a :class:`BayesianNetwork`.

    ``strict`` (default) raises :class:`DiagnosticError` if there is any error
    diagnostic; ``strict=False`` builds what it can and leaves the diagnostics on
    the returned network's ``pcore_diagnostics`` attribute.

    ``path`` is the script's own file path, if any; a relative ``include``
    (Phase 4) resolves against its directory. Without it, includes resolve
    against the current working directory.
    """
    diags = Diagnostics(list(program.diagnostics))
    model = program.model
    if model is None:
        if strict:
            raise DiagnosticError(diags.items, program.source)
        return None

    base_dir = os.path.dirname(os.path.abspath(path)) if path else None
    bn = _lower_model(model, diags, base_dir=base_dir)
    all_diags = diags.sorted
    if strict and any(d.is_error for d in all_diags):
        raise DiagnosticError(all_diags, program.source)
    if bn is not None:
        bn.pcore_diagnostics = all_diags
    return bn


def compile_script(source: str, strict: bool = True, path: str | None = None):
    """Parse and lower a PCore script in one call."""
    return to_network(parse(source), strict=strict, path=path)


def check(source: str, path: str | None = None):
    """Parse + lower leniently and return every diagnostic (parse and lowering)."""
    program = parse(source)
    diags = Diagnostics(list(program.diagnostics))
    model = program.model
    if model is not None:
        base_dir = os.path.dirname(os.path.abspath(path)) if path else None
        _lower_model(model, diags, base_dir=base_dir)
    return diags.sorted
