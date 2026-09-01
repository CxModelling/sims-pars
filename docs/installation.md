# Installation

## From PyPI

```bash
pip install sims-pars
```

`sims-pars` supports **Python 3.10 – 3.14**.

## Optional extras

| Extra | Installs | Enables |
|-------|----------|---------|
| `plot` | matplotlib | `BayesianNetwork.plot()` and notebook figures |
| `hme`  | gpflow, tensorflow | history-matching emulator (`sims_pars.fit.hme`) |
| `docs` | mkdocs-material, mkdocstrings, mkdocs-jupyter | building this site |
| `dev`  | pytest, pytest-cov, ruff | the test suite and linter |

```bash
pip install "sims-pars[plot]"
```

## Development setup

```bash
git clone https://github.com/CxModelling/sims-pars
cd sims-pars
pip install -e ".[dev,plot]"
pytest
ruff check src tests
```

The project uses a `src/` layout and a single `pyproject.toml`; there is no
`setup.py` or `requirements/` directory.

### Building the docs locally

```bash
pip install -e ".[docs]"
python scripts/stage_tutorials.py   # copies notebooks/ into docs/tutorials/
mkdocs serve
```
