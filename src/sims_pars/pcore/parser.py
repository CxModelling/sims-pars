"""Recursive-descent parser: tokens -> Program.

``parse`` never raises. A malformed statement produces a diagnostic anchored at
its span and the parser recovers to the next line, so one bad line never stops
the parse and nothing is ever silently dropped.
"""
from __future__ import annotations

import re

from sims_pars.pcore.ast import IncludeDef, Model, NodeDef
from sims_pars.pcore.diagnostics import Diagnostic, Diagnostics
from sims_pars.pcore.lexer import Token, TokenKind, lex

__all__ = ['Program', 'parse']

_END = {TokenKind.NEWLINE, TokenKind.RBRACE, TokenKind.EOF}
_IDENT_RE = r'[A-Za-z_]\w*'


class Program:
    """The parsed result. Holds every model plus the full diagnostic list."""

    def __init__(self, models: list[Model], diagnostics: list[Diagnostic], source: str):
        self.models = models
        self.diagnostics = diagnostics
        self.source = source

    @property
    def model(self) -> Model | None:
        """The first model — the one the legacy ``bayes_net_from_script`` returns."""
        return self.models[0] if self.models else None

    @property
    def errors(self) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.is_error]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_network(self, strict: bool = True):
        from sims_pars.pcore.lower import to_network
        return to_network(self, strict=strict)

    def render_diagnostics(self) -> str:
        return "\n".join(d.render(self.source) for d in self.diagnostics)


class _Parser:
    def __init__(self, tokens: list[Token], diags: Diagnostics, source: str):
        self.toks = tokens
        self.pos = 0
        self.diags = diags
        self.source = source

    # --- token helpers ---------------------------------------------------
    def peek(self, k: int = 0) -> Token:
        j = min(self.pos + k, len(self.toks) - 1)
        return self.toks[j]

    def at(self, *kinds: TokenKind) -> bool:
        return self.peek().kind in kinds

    def advance(self) -> Token:
        t = self.toks[self.pos]
        if self.pos < len(self.toks) - 1:
            self.pos += 1
        return t

    def skip_newlines(self) -> None:
        while self.at(TokenKind.NEWLINE):
            self.advance()

    def recover_line(self) -> None:
        while not self.at(TokenKind.NEWLINE, TokenKind.RBRACE, TokenKind.EOF):
            self.advance()

    # --- grammar -------------------------------------------------------
    def parse_program(self) -> list[Model]:
        models: list[Model] = []
        self.skip_newlines()
        while not self.at(TokenKind.EOF):
            if self.at(TokenKind.PCORE):
                models.append(self.parse_model())
                self.skip_newlines()
            else:
                t = self.advance()
                self.diags.error(
                    t.span, "E0100",
                    f"expected a 'PCore' block, found {t.text!r}",
                    hint="a script is 'PCore <Name> { ... }'",
                )
                self.skip_newlines()
        if not models:
            eof = self.toks[-1]
            self.diags.error(eof.span, "E0101", "no 'PCore' block in the script")
        return models

    def parse_model(self) -> Model:
        kw = self.advance()  # PCORE
        if not self.at(TokenKind.IDENT):
            bad = self.peek()
            self.diags.error(bad.span, "E0102", "expected a model name after 'PCore'")
            name, name_span = "?", bad.span
        else:
            nt = self.advance()
            name, name_span = nt.text, nt.span

        if self.at(TokenKind.LBRACE):
            self.advance()
        else:
            self.diags.error(self.peek().span, "E0103", "expected '{' to open the model")

        model = Model(name=name, name_span=name_span, span=kw.span.to(name_span))
        while True:
            self.skip_newlines()
            if self.at(TokenKind.RBRACE):
                self.advance()
                break
            if self.at(TokenKind.EOF):
                self.diags.error(self.peek().span, "E0104",
                                 f"model {name!r} is never closed with '}}'")
                break
            if self.at(TokenKind.PCORE):
                break  # a new model — let the program loop take it
            if self.at(TokenKind.FOR):
                model.statements.extend(self.parse_plate())
                continue
            if self.at(TokenKind.INCLUDE):
                inc = self.parse_include()
                if inc is not None:
                    model.includes.append(inc)
                continue
            stmt = self.parse_statement()
            if stmt is not None:
                model.statements.append(stmt)
        return model

    def parse_statement(self) -> NodeDef | None:
        first = self.peek()
        if not self.at(TokenKind.IDENT):
            self.diags.error(first.span, "E0110",
                             f"expected a node name, found {first.text or 'end of line'!r}")
            self.recover_line()
            return None

        name_tok = self.advance()
        name, name_span = name_tok.text, name_tok.span

        type_ann, type_span = None, None
        if self.at(TokenKind.COLON):
            self.advance()
            if self.at(TokenKind.IDENT):
                t = self.advance()
                type_ann, type_span = t.text, t.span
            else:
                bad = self.peek()
                self.diags.error(bad.span, "E0111",
                                 f"expected a type name after ':', found {bad.text or 'end of line'!r}",
                                 hint="e.g. 'x: float ~ norm(0, 1)'")
                self.recover_line()
                return None

        if self.at(TokenKind.TILDE, TokenKind.EQUALS):
            op_tok = self.advance()
            op = op_tok.text
            if not self.at(TokenKind.EXPR):
                self.diags.error(op_tok.span, "E0112",
                                 f"expected an expression after {op!r}")
                self.recover_line()
                return None
            rhs_tok = self.advance()
            rhs, rhs_span = rhs_tok.text, rhs_tok.span
        elif self.at(*_END, TokenKind.DESCRIPTION):
            op, rhs, rhs_span = "", None, None
        else:
            bad = self.peek()
            self.diags.error(bad.span, "E0113",
                             f"expected '~', '=' or end of line, found {bad.text!r}")
            self.recover_line()
            return None

        description = None
        if self.at(TokenKind.DESCRIPTION):
            description = self.advance().text or None

        end = self.peek()
        if not self.at(*_END):
            self.diags.error(end.span, "E0114",
                             f"unexpected {end.text!r} after the statement")
            self.recover_line()

        span = name_span.to(rhs_span or name_span)
        return NodeDef(name=name, name_span=name_span, op=op, rhs=rhs,
                       rhs_span=rhs_span, span=span, description=description,
                       type_ann=type_ann, type_span=type_span)

    # --- Phase 4: plates -------------------------------------------------
    def _expect_int(self, code: str, message: str) -> int | None:
        if not self.at(TokenKind.INT):
            self.diags.error(self.peek().span, code, message)
            self.recover_line()
            return None
        return int(self.advance().text)

    def parse_plate(self) -> list[NodeDef]:
        """``for i in lo..hi { statement* }`` -> lo..hi copies of the body,
        with ``i`` substituted (see :func:`_expand_plate`)."""
        kw = self.advance()  # FOR

        if not self.at(TokenKind.IDENT):
            self.diags.error(self.peek().span, "E0120",
                             "expected a loop variable after 'for'")
            self.recover_line()
            return []
        var = self.advance().text

        if not self.at(TokenKind.IN):
            self.diags.error(self.peek().span, "E0121",
                             "expected 'in' after the loop variable",
                             hint="e.g. 'for i in 1..5 { ... }'")
            self.recover_line()
            return []
        self.advance()

        lo = self._expect_int("E0122", "expected an integer lower bound")
        if lo is None:
            return []
        if not self.at(TokenKind.DOTDOT):
            self.diags.error(self.peek().span, "E0123",
                             "expected '..' between the plate bounds")
            self.recover_line()
            return []
        self.advance()
        hi = self._expect_int("E0124", "expected an integer upper bound")
        if hi is None:
            return []

        if self.at(TokenKind.LBRACE):
            self.advance()
        else:
            self.diags.error(self.peek().span, "E0125",
                             "expected '{' to open the plate body")
            return []

        body: list[NodeDef] = []
        while True:
            self.skip_newlines()
            if self.at(TokenKind.RBRACE):
                self.advance()
                break
            if self.at(TokenKind.EOF, TokenKind.PCORE):
                self.diags.error(self.peek().span, "E0126",
                                 f"plate 'for {var} in ...' is never closed with '}}'")
                break
            if self.at(TokenKind.FOR):
                body.extend(self.parse_plate())
                continue
            if self.at(TokenKind.INCLUDE):
                self.diags.error(self.peek().span, "E0127",
                                 "'include' is not allowed inside a plate")
                self.advance()
                if self.at(TokenKind.EXPR):
                    self.advance()
                continue
            stmt = self.parse_statement()
            if stmt is not None:
                body.append(stmt)

        if hi < lo:
            self.diags.warning(kw.span, "E0128", f"empty plate range: {lo}..{hi}")
        return _expand_plate(var, lo, hi, body)

    # --- Phase 4: composition ---------------------------------------------
    def parse_include(self) -> IncludeDef | None:
        kw = self.advance()  # INCLUDE

        if not self.at(TokenKind.EXPR):
            self.diags.error(kw.span, "E0130",
                             "expected a quoted path after 'include'",
                             hint='e.g. include "shared.pcore"')
            self.recover_line()
            return None
        tok = self.advance()
        text = tok.text.strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
            path = text[1:-1]
        else:
            self.diags.error(tok.span, "E0131",
                             "the include path must be a quoted string",
                             hint='e.g. include "shared.pcore"')
            self.recover_line()
            return None

        end = self.peek()
        if not self.at(*_END):
            self.diags.error(end.span, "E0132",
                             f"unexpected {end.text!r} after include")
            self.recover_line()
        return IncludeDef(path=path, span=kw.span.to(tok.span))


def _subst(text: str | None, var: str, k: int) -> str | None:
    """Substitute a plate's loop variable inside a raw RHS string, leaving
    quoted contents untouched.

    Two forms are recognised outside of string literals:

    * ``name[var]`` (a sibling plated node, indexed by the loop variable) ->
      ``name_k``
    * a bare, whole-word ``var`` (the loop variable used as a value) -> ``k``
    """
    if not text:
        return text

    bracket_idx = re.compile(rf'\b({_IDENT_RE})\[{re.escape(var)}\]')
    bare = re.compile(rf'\b{re.escape(var)}\b')

    segments: list[tuple[bool, str]] = []
    buf: list[str] = []
    quote = ""
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if quote:
            buf.append(c)
            if c == "\\" and i + 1 < n:
                buf.append(text[i + 1])
                i += 2
                continue
            if c == quote:
                segments.append((True, "".join(buf)))
                buf = []
                quote = ""
            i += 1
            continue
        if c in "\"'":
            if buf:
                segments.append((False, "".join(buf)))
                buf = []
            quote = c
            buf.append(c)
            i += 1
            continue
        buf.append(c)
        i += 1
    if buf:
        segments.append((quote != "", "".join(buf)))

    out = []
    for is_quoted, chunk in segments:
        if not is_quoted:
            chunk = bracket_idx.sub(lambda m: f'{m.group(1)}_{k}', chunk)
            chunk = bare.sub(str(k), chunk)
        out.append(chunk)
    return "".join(out)


def _expand_plate(var: str, lo: int, hi: int, body: list[NodeDef]) -> list[NodeDef]:
    """Expand one plate body into ``lo..hi`` concrete :class:`NodeDef`\\ s.

    Declared names are suffixed (``x`` -> ``x_1, x_2, ...``); references of
    the form ``name[var]`` in a right-hand side pick the matching sibling;
    a bare ``var`` in a right-hand side becomes the literal iteration value.
    Nested plates are already expanded by the time this runs (parsing is
    depth-first), so this never needs to recurse.
    """
    out: list[NodeDef] = []
    for k in range(lo, hi + 1):
        for s in body:
            out.append(NodeDef(
                name=f'{s.name}_{k}',
                name_span=s.name_span,
                op=s.op,
                rhs=_subst(s.rhs, var, k),
                rhs_span=s.rhs_span,
                span=s.span,
                description=s.description,
                type_ann=s.type_ann,
                type_span=s.type_span,
            ))
    return out


def parse(source: str) -> Program:
    tokens, diags = lex(source)
    parser = _Parser(tokens, diags, source)
    models = parser.parse_program()
    return Program(models, diags.sorted, source)
