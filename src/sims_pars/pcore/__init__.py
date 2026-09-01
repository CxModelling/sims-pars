"""PCore — the sims-pars model-description language.

A real front end (lexer, parser, diagnostics) for the ``PCore { ... }`` scripts
that :func:`sims_pars.bayes_net_from_script` has always accepted. Compared with
the legacy regex parser it

* reports every problem with a source span instead of crashing with a raw
  Python exception or silently skipping the line,
* tokenises the input rather than deleting every space first, and
* keeps the resulting :class:`~sims_pars.bayesnet.bn.BayesianNetwork` byte-identical
  to the legacy parser for every script in ``tests/pcore/corpus``.

Public API::

    from sims_pars.pcore import parse, compile_script

    program = parse(text)          # never raises; program.diagnostics holds errors
    bn = compile_script(text)      # -> BayesianNetwork (strict: raises on error)
    bn = compile_script(text, strict=False)   # legacy-lenient: drop bad lines
"""
from sims_pars.pcore.diagnostics import Diagnostic, DiagnosticError, Severity, Span
from sims_pars.pcore.evaluator import EvalError, evaluate
from sims_pars.pcore.lower import check, compile_script, to_network
from sims_pars.pcore.parser import Program, parse

__all__ = [
    'parse',
    'Program',
    'compile_script',
    'check',
    'to_network',
    'Diagnostic',
    'DiagnosticError',
    'Severity',
    'Span',
    'evaluate',
    'EvalError',
]
