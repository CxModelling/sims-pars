from sims_pars.pcore.lexer import TokenKind, lex


def kinds(src):
    return [t.kind for t in lex(src)[0]]


def test_header_and_simple_statements():
    toks, diags = lex("PCore M {\n a = 1\n b ~ norm(0, 1)\n}\n")
    assert not diags
    seq = [t.kind for t in toks]
    assert seq[0] is TokenKind.PCORE
    assert TokenKind.LBRACE in seq and TokenKind.RBRACE in seq
    assert TokenKind.TILDE in seq and TokenKind.EQUALS in seq


def test_rhs_is_one_raw_token_with_spaces_preserved():
    toks, _ = lex("PCore M {\n b ~ norm(a,  1)\n}\n")
    exprs = [t.text for t in toks if t.kind is TokenKind.EXPR]
    assert exprs == ["norm(a,  1)"]


def test_rhs_is_bracket_aware():
    toks, _ = lex('PCore M {\n g ~ cat({"low risk": 4})\n}\n')
    exprs = [t.text for t in toks if t.kind is TokenKind.EXPR]
    assert exprs == ['cat({"low risk": 4})']
    assert sum(1 for t in toks if t.kind is TokenKind.RBRACE) == 1


def test_description_after_statement():
    toks, _ = lex("PCore M {\n a = 1 # the note, with commas\n}\n")
    desc = [t.text for t in toks if t.kind is TokenKind.DESCRIPTION]
    assert desc == ["the note, with commas"]


def test_line_comments():
    for c in ("#", "//", "--"):
        toks, diags = lex(f"PCore M {{\n {c} a comment\n a = 1\n}}\n")
        assert not diags
        idents = [t.text for t in toks if t.kind is TokenKind.IDENT]
        assert idents == ["M", "a"]


def test_leading_digit_is_not_an_identifier():
    toks, _ = lex("PCore M {\n 2 = x\n}\n")
    assert not any(t.kind is TokenKind.IDENT and t.text == "2" for t in toks)


def test_unterminated_string_is_a_diagnostic():
    _, diags = lex('PCore M {\n a = "oops\n}\n')
    assert any(d.code == "E0001" for d in diags)
