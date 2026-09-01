"""Every crash / silent-corruption case from the fault analysis now produces a
located diagnostic instead."""
import pytest

from sims_pars.pcore import DiagnosticError, Severity, check, compile_script, parse


def diags(src):
    return check(src)


def codes(src):
    return {d.code for d in diags(src)}


def wrap(body):
    return "PCore M {\n" + body + "\n}\n"


class TestNoMoreCrashes:
    def test_leading_hash_comment_is_not_a_crash(self):
        p = parse(wrap("# just a note\n    a = 1"))
        assert p.ok
        bn = p.to_network()
        assert bn.Order == ["a"]

    def test_slash_and_dash_comments(self):
        p = parse(wrap("// a comment\n-- another\n    a = 1"))
        assert p.ok and p.to_network().Order == ["a"]

    def test_unknown_distribution_suggests(self):
        d = diags(wrap("p ~ nrom(0, 1)"))
        assert d and d[0].code == "E0211"
        assert "norm" in (d[0].hint or "")

    def test_distribution_missing_required_arg(self):
        d = diags(wrap("c ~ cat()"))
        assert d and d[0].is_error
        assert d[0].code in {"E0212", "E0213"}

    def test_stray_tilde(self):
        d = diags(wrap("a = 1\n    b = 2\n    c = a ~ b"))
        assert any(x.is_error for x in d)

    def test_malformed_expression(self):
        d = diags(wrap("x = 2 +"))
        assert d and d[0].code == "E0221"

    def test_attribute_access_rejected_with_span(self):
        d = diags(wrap("x = 1\n    y = x.real"))
        assert d and d[0].is_error
        assert d[0].span.line == 3

    def test_unknown_function_is_not_a_crash(self):
        # FunctionLoci accepts it at parse time (legacy parity); the value is a
        # function node whose call target is unresolved — surfaced later, never a crash.
        p = parse(wrap("k = 3\n    y = expo(k)"))
        assert p.ok  # no parse error; matches legacy structure

    def test_leading_digit_name_is_rejected(self):
        d = diags(wrap("2 = x"))
        assert any(x.is_error for x in d)


class TestNoMoreSilentSkips:
    def test_unclassifiable_line_is_reported(self):
        d = diags(wrap("a = 1\n    a.b = 1\n    c = 2"))
        assert any(x.is_error for x in d)

    def test_strict_mode_refuses_to_build(self):
        with pytest.raises(DiagnosticError):
            compile_script(wrap("p ~ nrom(0, 1)"), strict=True)

    def test_lenient_mode_drops_the_bad_node(self):
        bn = compile_script(wrap("a = 1\n    b ~ nrom(0,1)\n    c = a + 1"),
                            strict=False)
        assert "a" in bn.Order and "c" in bn.Order
        assert any(x.is_error for x in bn.pcore_diagnostics)


class TestStringsKeepSpaces:
    def test_category_key_with_space_is_preserved(self):
        bn = compile_script(wrap('g ~ cat({"low risk": 4, "high risk": 1})'))
        keys = list(bn["g"].get_distribution().kv)
        assert keys == ["low risk", "high risk"]


def test_diagnostic_render_points_at_the_source():
    d = diags(wrap("p ~ nrom(0, 1)"))[0]
    rendered = d.render("PCore M {\np ~ nrom(0, 1)\n}\n")
    assert "2:" in rendered and "^" in rendered
    assert d.severity is Severity.ERROR
