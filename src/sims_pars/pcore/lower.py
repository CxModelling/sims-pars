"""Lowering: a parsed :class:`Program` -> :class:`BayesianNetwork`.

Phase 1 reuses the legacy loci classes and ``BayesianNetwork.append_loci``, so a
valid script produces a byte-identical network. The value it adds is that every
failure that used to be a raw exception (unknown distribution tag, malformed
expression, a cycle) becomes a diagnostic with a source span.
"""
from __future__ import annotations

import difflib
import re

from sims_pars.pcore.ast import Model, NodeDef
from sims_pars.pcore.diagnostics import DiagnosticError, Diagnostics
from sims_pars.pcore.parser import Program, parse

__all__ = ['to_network', 'compile_script']

_LEADING_IDENT = re.compile(r"[A-Za-z_]\w*")


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


def _lower_model(model: Model, diags: Diagnostics):
    from sims_pars.bayesnet import BayesianNetwork
    from sims_pars.bayesnet.loci import (
        DistributionLoci,
        ExoValueLoci,
        FunctionLoci,
        PseudoLoci,
        ValueLoci,
    )

    bn = BayesianNetwork(model.name)
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


def to_network(program: Program, strict: bool = True):
    """Lower a parsed program to a :class:`BayesianNetwork`.

    ``strict`` (default) raises :class:`DiagnosticError` if there is any error
    diagnostic; ``strict=False`` builds what it can and leaves the diagnostics on
    the returned network's ``pcore_diagnostics`` attribute.
    """
    diags = Diagnostics(list(program.diagnostics))
    model = program.model
    if model is None:
        if strict:
            raise DiagnosticError(diags.items, program.source)
        return None

    bn = _lower_model(model, diags)
    all_diags = diags.sorted
    if strict and any(d.is_error for d in all_diags):
        raise DiagnosticError(all_diags, program.source)
    if bn is not None:
        bn.pcore_diagnostics = all_diags
    return bn


def compile_script(source: str, strict: bool = True):
    """Parse and lower a PCore script in one call."""
    return to_network(parse(source), strict=strict)


def check(source: str):
    """Parse + lower leniently and return every diagnostic (parse and lowering)."""
    program = parse(source)
    diags = Diagnostics(list(program.diagnostics))
    model = program.model
    if model is not None:
        _lower_model(model, diags)
    return diags.sorted
