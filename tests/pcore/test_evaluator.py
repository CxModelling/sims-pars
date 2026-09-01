"""The dedicated Phase-3 expression evaluator (no eval())."""
import pytest

from sims_pars.pcore.evaluator import EvalError, evaluate, free_names


@pytest.mark.parametrize('expr, expected', [
    ('1 + 2 * 3', 7),
    ('(1 + 2) * 3', 9),
    ('2 ** 3 ** 2', 512),          # right-associative
    ('-2 ** 2', -4),               # '**' binds tighter than unary '-': -(2 ** 2)
    ('10 // 3', 3),
    ('10 % 3', 1),
    ('1 < 2 < 3', True),           # chained comparison
    ('1 < 2 < 0', False),
    ('3 if True else 4', 3),
    ('3 if False else 4', 4),
    ('True and False', False),
    ('True or False', True),
    ('[1, 2, 3]', [1, 2, 3]),
    ('(1, 2)', (1, 2)),
    ('{1, 2, 2}', {1, 2}),
    ('{"a": 1}', {'a': 1}),
    ('min(3, 1, 2)', 1),
    ('exp(0)', 1.0),
])
def test_evaluates_the_operator_table(expr, expected):
    assert evaluate(expr) == expected


def test_undefined_name_raises_eval_error_not_python_nameerror():
    with pytest.raises(EvalError):
        evaluate('x + 1')


def test_env_supplies_names():
    assert evaluate('x + 1', {'x': 4}) == 5


@pytest.mark.parametrize('expr', [
    '__import__("os")',
    'os.system("echo hi")',
    '(lambda: 1)()',
    '[x for x in range(3)]',
    'open("x")',
])
def test_rejects_disallowed_constructs(expr):
    with pytest.raises(EvalError):
        evaluate(expr)


def test_free_names_excludes_known_functions():
    assert free_names('exp(a) + b') == {'a', 'b'}
    assert free_names('1 + 2') == set()
