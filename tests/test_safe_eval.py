import pytest

from sims_pars.util import MATH_FUNC, UnsafeExpression, safe_eval


class TestSafeEvalAccepts:
    def test_arithmetic(self):
        assert safe_eval('1 + 2 * 3') == 7

    def test_local_names(self):
        assert safe_eval('x + y', local={'x': 2, 'y': 3}) == 5

    def test_math_func(self):
        assert safe_eval('exp(0)', MATH_FUNC) == 1.0

    def test_safe_builtin(self):
        assert safe_eval('max(a, b)', local={'a': 1, 'b': 9}) == 9

    def test_container_literals(self):
        assert safe_eval('{"a": 1}') == {'a': 1}
        assert safe_eval('[1, 2, 3]') == [1, 2, 3]

    def test_conditional(self):
        assert safe_eval('a if a > 0 else -a', local={'a': -4}) == 4


class TestSafeEvalRejects:
    @pytest.mark.parametrize('expr', [
        '__import__("os").system("echo hi")',
        '().__class__.__bases__',
        '[c for c in range(3)]',
        'lambda: 1',
        'x.attribute',
        'data[0]',
        '(1).__class__',
    ])
    def test_disallowed(self, expr):
        with pytest.raises((UnsafeExpression, SyntaxError)):
            safe_eval(expr, local={'x': 1, 'data': [1]})

    def test_builtins_are_stripped(self):
        with pytest.raises(NameError):
            safe_eval('open("x")')
