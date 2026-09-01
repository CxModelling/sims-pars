"""Source spans and diagnostics — the shared error currency of the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

__all__ = ['Span', 'Severity', 'Diagnostic', 'DiagnosticError', 'Diagnostics']


@dataclass(frozen=True)
class Span:
    """A half-open ``[start, end)`` slice of the source, 0-based offsets, with
    1-based line/column for display."""
    start: int
    end: int
    line: int
    col: int

    @staticmethod
    def zero() -> "Span":
        return Span(0, 0, 1, 1)

    def to(self, other: "Span") -> "Span":
        return Span(self.start, other.end, self.line, self.col)

    def __str__(self) -> str:
        return f"{self.line}:{self.col}"


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(frozen=True)
class Diagnostic:
    span: Span
    severity: Severity
    code: str
    message: str
    hint: str | None = None

    @property
    def is_error(self) -> bool:
        return self.severity is Severity.ERROR

    def render(self, source: str | None = None) -> str:
        loc = f"{self.span.line}:{self.span.col}"
        head = f"{loc}: {self.severity} [{self.code}] {self.message}"
        if source is None:
            return head + (f"\n    hint: {self.hint}" if self.hint else "")
        lines = source.splitlines()
        out = [head]
        if 1 <= self.span.line <= len(lines):
            src_line = lines[self.span.line - 1]
            out.append(f"  {self.span.line:>4} | {src_line}")
            caret_pad = " " * (self.span.col - 1)
            width = max(1, min(self.span.end - self.span.start, len(src_line)))
            out.append(f"       | {caret_pad}{'^' * width}")
        if self.hint:
            out.append(f"    hint: {self.hint}")
        return "\n".join(out)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.render()


class DiagnosticError(Exception):
    """Raised by strict entry points when the program has error diagnostics."""

    def __init__(self, diagnostics: list[Diagnostic], source: str | None = None):
        self.diagnostics = list(diagnostics)
        self.source = source
        errs = [d for d in self.diagnostics if d.is_error]
        shown = "\n".join(d.render(source) for d in errs[:5])
        more = "" if len(errs) <= 5 else f"\n... and {len(errs) - 5} more"
        super().__init__(f"{len(errs)} error(s) in PCore script:\n{shown}{more}")


@dataclass
class Diagnostics:
    """A growing list of diagnostics, sorted by position on read."""
    items: list[Diagnostic] = field(default_factory=list)

    def add(self, span: Span, severity: Severity, code: str, message: str,
            hint: str | None = None) -> None:
        self.items.append(Diagnostic(span, severity, code, message, hint))

    def error(self, span: Span, code: str, message: str, hint: str | None = None) -> None:
        self.add(span, Severity.ERROR, code, message, hint)

    def warning(self, span: Span, code: str, message: str, hint: str | None = None) -> None:
        self.add(span, Severity.WARNING, code, message, hint)

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.items if d.is_error]

    @property
    def sorted(self) -> list[Diagnostic]:
        return sorted(self.items, key=lambda d: (d.span.start, d.span.end))

    def __bool__(self) -> bool:
        return bool(self.items)

    def __iter__(self):
        return iter(self.sorted)

    def __len__(self) -> int:
        return len(self.items)
