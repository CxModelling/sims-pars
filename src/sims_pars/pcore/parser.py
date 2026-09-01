"""Recursive-descent parser: tokens -> Program.

``parse`` never raises. A malformed statement produces a diagnostic anchored at
its span and the parser recovers to the next line, so one bad line never stops
the parse and nothing is ever silently dropped.
"""
from __future__ import annotations

from sims_pars.pcore.ast import Model, NodeDef
from sims_pars.pcore.diagnostics import Diagnostic, Diagnostics
from sims_pars.pcore.lexer import Token, TokenKind, lex

__all__ = ['Program', 'parse']

_END = {TokenKind.NEWLINE, TokenKind.RBRACE, TokenKind.EOF}


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

        if self.at(TokenKind.COLON):
            self.advance()
            self.recover_line()
            self.diags.error(name_span.to(self.peek().span), "E0111",
                             "type annotations are not supported yet",
                             hint="drop the ': type' for now")
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
                       rhs_span=rhs_span, span=span, description=description)


def parse(source: str) -> Program:
    tokens, diags = lex(source)
    parser = _Parser(tokens, diags, source)
    models = parser.parse_program()
    return Program(models, diags.sorted, source)
