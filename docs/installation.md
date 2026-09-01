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
| `hme`  | gpytorch, torch (CPU) | history-matching GP emulator (`sims_pars.fit.hme`) |
| `docs` | mkdocs-material, mkdocstrings, mkdocs-jupyter | building this site |
| `dev`  | pytest, pytest-cov, ruff | the test suite and linter |

```bash
pip install "sims-pars[plot]"
```

## From GitHub

Install the latest `main` directly, without cloning:

```bash
pip install "git+https://github.com/CxModelling/sims-pars.git"
```

Pin a released version or any branch / commit with `@<ref>`, and add extras in
the usual brackets:

```bash
pip install "sims-pars[plot] @ git+https://github.com/CxModelling/sims-pars.git@v3.0.0"
pip install "git+https://github.com/CxModelling/sims-pars.git@main#egg=sims-pars[hme]"
```

`uv` works the same way:

```bash
uv pip install "git+https://github.com/CxModelling/sims-pars.git@v3.0.0"
```

In a `pyproject.toml` / `requirements.txt`:

```
sims-pars @ git+https://github.com/CxModelling/sims-pars.git@v3.0.0
```

Or download a source tarball for a tag and install it offline:

```bash
pip install https://github.com/CxModelling/sims-pars/archive/refs/tags/v3.0.0.tar.gz
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
