"""PCore abstract syntax — Phase 1 surface (v1-equivalent).

One statement type, :class:`NodeDef`, covers all five legacy loci kinds; which
one it lowers to is decided by ``op`` and the shape of ``rhs`` (never by
evaluating anything during parsing).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sims_pars.pcore.diagnostics import Span

__all__ = ['NodeDef', 'IncludeDef', 'Model']


@dataclass
class NodeDef:
    name: str
    name_span: Span
    op: str                      # '~' random | '=' deterministic | '' exogenous
    rhs: str | None              # raw right-hand-side source (whitespace preserved)
    rhs_span: Span | None
    span: Span
    description: str | None = None
    type_ann: str | None = None  # Phase 3: 'name : type' annotation, e.g. 'float'
    type_span: Span | None = None

    @property
    def is_sample(self) -> bool:
        return self.op == '~'

    @property
    def is_assign(self) -> bool:
        return self.op == '='

    @property
    def is_exogenous(self) -> bool:
        return self.op == ''


@dataclass
class IncludeDef:
    """Phase 4: ``include "path.pcore"`` — splice another file's nodes in."""
    path: str
    span: Span


@dataclass
class Model:
    name: str
    name_span: Span
    span: Span
    statements: list[NodeDef] = field(default_factory=list)
    includes: list[IncludeDef] = field(default_factory=list)
