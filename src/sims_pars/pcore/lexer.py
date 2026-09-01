"""Tokeniser for PCore scripts.

Coarse by design: statement structure is tokenised precisely, but the
right-hand side of ``~`` / ``=`` is captured as one raw ``EXPR`` token whose text
is handed to the existing distribution / expression machinery during lowering.
That keeps the output identical to the legacy parser while still giving every
statement a real source span.

Whitespace is *not* stripped globally. Strings keep their contents verbatim, and
``#`` / ``//`` / ``--`` at the start of a logical line are comments rather than a
crash or a silently dropped node.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from sims_pars.pcore.diagnostics import Diagnostics, Span

__all__ = ['TokenKind', 'Token', 'lex']

_STRUCT = {"{": "LBRACE", "}": "RBRACE", "~": "TILDE", "=": "EQUALS", ":": "COLON"}


class TokenKind(Enum):
    PCORE = auto()
    FOR = auto()            # Phase 4: plates
    IN = auto()
    INCLUDE = auto()        # Phase 4: composition
    IDENT = auto()
    INT = auto()             # Phase 4: plate bounds
    DOTDOT = auto()          # Phase 4: '..' range operator
    LBRACE = auto()
    RBRACE = auto()
    TILDE = auto()
    EQUALS = auto()
    COLON = auto()
    EXPR = auto()          # raw right-hand-side source
    DESCRIPTION = auto()   # text after '#'
    NEWLINE = auto()
    EOF = auto()


# Reserved words, matched case-insensitively like 'PCore'. None of them are
# used as identifiers anywhere in the v1 corpus.
_KEYWORDS = {
    "pcore": TokenKind.PCORE,
    "for": TokenKind.FOR,
    "in": TokenKind.IN,
    "include": TokenKind.INCLUDE,
}


@dataclass(frozen=True)
class Token:
    kind: TokenKind
    text: str
    span: Span

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Token({self.kind.name}, {self.text!r}, {self.span})"


_IDENT_START = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
_IDENT_CONT = _IDENT_START | set("0123456789")


class _Cursor:
    def __init__(self, src: str):
        self.src = src
        self.i = 0
        self.line = 1
        self.col = 1

    def eof(self) -> bool:
        return self.i >= len(self.src)

    def peek(self, k: int = 0) -> str:
        j = self.i + k
        return self.src[j] if j < len(self.src) else ""

    def advance(self) -> str:
        ch = self.src[self.i]
        self.i += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def span_from(self, start_i: int, start_line: int, start_col: int) -> Span:
        return Span(start_i, self.i, start_line, start_col)


def lex(src: str) -> tuple[list[Token], Diagnostics]:
    cur = _Cursor(src)
    diags = Diagnostics()
    toks: list[Token] = []
    at_line_start = True   # only-whitespace seen on the current logical line
    expect_rhs = False     # the previous significant token was '~' or '='

    def emit(kind: TokenKind, text: str, span: Span) -> None:
        toks.append(Token(kind, text, span))

    def read_expr_run() -> tuple[str, Span, bool]:
        """Consume raw RHS source to end of line, or a top-level '}' / '#'.

        Bracket-aware: a '}' or '#' inside ``()``/``[]``/``{}`` is part of the
        expression (e.g. ``cat({"a": 1})``), not a model close or a description.
        """
        si, sl, sc = cur.i, cur.line, cur.col
        quote = ""
        depth = 0
        while not cur.eof():
            c = cur.peek()
            if quote:
                if c == "\\" and cur.peek(1):
                    cur.advance(); cur.advance(); continue
                if c == quote:
                    quote = ""
                cur.advance(); continue
            if c in "\"'":
                quote = c
                cur.advance(); continue
            if c == "\n":
                break
            if c in "([{":
                depth += 1
            elif c in ")]}":
                if c == "}" and depth <= 0:
                    break
                depth -= 1
            elif c == "#" and depth <= 0:
                break
            cur.advance()
        return src[si:cur.i].rstrip(), cur.span_from(si, sl, sc), bool(quote)

    while not cur.eof():
        ch = cur.peek()
        si, sl, sc = cur.i, cur.line, cur.col

        if ch in " \t\r":
            cur.advance()
            continue

        if ch == "\n":
            cur.advance()
            emit(TokenKind.NEWLINE, "\n", cur.span_from(si, sl, sc))
            at_line_start = True
            expect_rhs = False
            continue

        # line comments: only at the start of a logical line
        if at_line_start and (ch == "#" or src[cur.i:cur.i + 2] in ("//", "--")):
            while not cur.eof() and cur.peek() != "\n":
                cur.advance()
            continue

        if expect_rhs:
            text, span, unterminated = read_expr_run()
            if unterminated:
                diags.error(span, "E0001", "unterminated string literal")
            if text:
                emit(TokenKind.EXPR, text, span)
            expect_rhs = False
            at_line_start = False
            continue

        # trailing description: '#' after some content runs to end of line
        if ch == "#":
            cur.advance()
            while not cur.eof() and cur.peek() in " \t":
                cur.advance()
            ti = cur.i
            while not cur.eof() and cur.peek() != "\n":
                cur.advance()
            emit(TokenKind.DESCRIPTION, src[ti:cur.i].rstrip(),
                 cur.span_from(si, sl, sc))
            continue

        # Phase 4: '..' range operator, checked ahead of the single-char dispatch
        if ch == "." and cur.peek(1) == ".":
            cur.advance()
            cur.advance()
            emit(TokenKind.DOTDOT, "..", cur.span_from(si, sl, sc))
            at_line_start = False
            continue

        if ch in _STRUCT:
            cur.advance()
            emit(TokenKind[_STRUCT[ch]], ch, cur.span_from(si, sl, sc))
            at_line_start = False
            expect_rhs = ch in "~="
            continue

        if ch in _IDENT_START:
            while not cur.eof() and cur.peek() in _IDENT_CONT:
                cur.advance()
            text = src[si:cur.i]
            kind = _KEYWORDS.get(text.lower(), TokenKind.IDENT)
            emit(kind, text, cur.span_from(si, sl, sc))
            at_line_start = False
            # 'include' takes a raw (quoted) path, captured the same way '~'/'='
            # capture their right-hand side.
            expect_rhs = kind is TokenKind.INCLUDE
            continue

        # Phase 4: plate bounds ('for i in 1..5')
        if ch.isdigit():
            while not cur.eof() and cur.peek().isdigit():
                cur.advance()
            emit(TokenKind.INT, src[si:cur.i], cur.span_from(si, sl, sc))
            at_line_start = False
            continue

        # a bare identifier line is exogenous; anything else here is junk the
        # parser will report against this span.
        text, span, unterminated = read_expr_run()
        if unterminated:
            diags.error(span, "E0001", "unterminated string literal")
        if text:
            emit(TokenKind.EXPR, text, span)
        at_line_start = False

    emit(TokenKind.EOF, "", Span(cur.i, cur.i, cur.line, cur.col))
    return toks, diags
