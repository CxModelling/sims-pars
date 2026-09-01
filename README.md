# sims-pars

[![CI](https://github.com/CxModelling/sims-pars/actions/workflows/ci.yml/badge.svg)](https://github.com/CxModelling/sims-pars/actions/workflows/ci.yml)
[![Docs](https://github.com/CxModelling/sims-pars/actions/workflows/docs.yml/badge.svg)](https://cxmodelling.github.io/sims-pars/)
[![Python](https://img.shields.io/badge/python-3.10%20%E2%80%93%203.14-blue)](https://pypi.org/project/sims-pars/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Coverage](https://img.shields.io/badge/coverage-74%25-yellowgreen)](#code-quality)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Serving stochastic parameters to simulation models.

`sims-pars` turns a compact text description of a probabilistic model — a *PCore
script* — into a Bayesian network you can sample from, condition on inputs, and
calibrate to data with approximate Bayesian computation.

```python
from sims_pars import bayes_net_from_script, sample

bn = bayes_net_from_script('''
PCore SIR {
    beta  ~ unif(1, 20)
    gamma ~ unif(0.1, 1)
    r0 = beta / gamma
}
''')

sample(bn)
```

## Install

```bash
pip install sims-pars
```

Python 3.10+ · NumPy 2 · SciPy · pandas · pydantic 2 · networkx.
Optional extras: `plot`, `hme`, `docs`, `dev`.

## Documentation

<https://cxmodelling.github.io/sims-pars/> — start with the three-chapter
[Getting Started](https://cxmodelling.github.io/sims-pars/getting-started/)
guide.

## Development

```bash
pip install -e ".[dev,plot]"
pytest
ruff check src tests
```

## Code quality

| Aspect | Status |
|---|---|
| **Tests** | 245 `pytest` tests, green on every supported Python. `tests/conftest.py` seeds the RNG so runs are deterministic. |
| **Coverage** | ~74% line coverage (`pytest --cov=sims_pars`). Highest on the parts most likely to break silently: `pcore` (lexer 98%, ast/cli 100%, evaluator 90%), `bayesnet`, `fn`, `util`'s `safe_eval`. Lowest on the legacy `simulation.parcore` actor tree and the unmaintained `sims_pars.data` package. |
| **CI** | [`ci.yml`](.github/workflows/ci.yml) runs `ruff check` + `pytest --cov` on a **Python 3.10 – 3.14** matrix, plus a strict `mkdocs build`, on every push and PR. |
| **Lint** | `ruff` (pyflakes `F` + pycodestyle errors `E9` + flake8-bugbear `B`) gates the whole tree; tests add the stricter rules. No suppressions outside `pyproject.toml`. |
| **Regression guard** | `tests/test_regression_np2.py` imports every public submodule so a NumPy-2 / pydantic-2 breakage surfaces as a test failure, not a runtime surprise. |
| **PCore compatibility oracle** | `tests/pcore/corpus/` holds 24 scripts harvested from every test, notebook and `__main__` block; `tests/pcore/test_corpus.py` asserts the new front end compiles each to a **byte-identical** `BayesianNetwork` as the legacy parser (one enumerated exception: a cyclic script the old parser accepted and the new one rejects). |
| **Security** | Script expressions are evaluated through `sims_pars.util.safe_eval` — `__builtins__` cleared, the AST restricted to an arithmetic / function-call allow-list. Attribute access, subscripting, comprehensions, lambdas and imports are rejected with a source span, not executed. `tests/test_safe_eval.py` covers the rejections. |
| **Known gaps** | `sims_pars.data.*` (imports the defunct `epidag`) and `sims_pars.fitting.*` are unmaintained — flagged with greppable `TODO(epidag)` / `TODO(chromosome-api)` markers and excluded from the import sweep. See the [changelog](docs/changelog.md#known-issues-annotated-with-todo-markers-not-fixed-here). |

Coverage figure is from the 3.12 CI job; re-measure with `pytest --cov=sims_pars --cov-report=term-missing`.
