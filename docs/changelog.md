# Changelog

## 3.0.0 (unreleased)

A modernisation release: the package now installs and passes its tests on
**Python 3.10 – 3.14**, on **NumPy 2** and **pydantic 2**. It also gains a real
parser for PCore scripts, a documentation site, and a CPU-only GP emulator that
drops the TensorFlow dependency.

### Breaking

- **Python 3.10+** required (was 3.9).
- **pydantic 2** required (was pydantic 1). `.schema()` → `.model_json_schema()`,
  `.dict()` → `.model_dump()`, `pydantic.types.*` → `Annotated[..., Field(...)]`,
  `class C(BaseModel, metaclass=ABCMeta)` → `class C(BaseModel, ABC)`.
- **NumPy 2** required. `np.Inf` / `np.NaN` (removed upstream) replaced with
  `np.inf` / `np.nan`.
- **Packaging** consolidated into a single `pyproject.toml` (setuptools, `src/`
  layout). `src/setup.py`, `requirements/`, `requirements.txt` and
  `environments/env_dev.yml` were removed — use `pip install -e ".[dev]"`. The
  unused `astunparse` dependency was dropped (`ast.unparse` is stdlib on 3.10+).
- Optional extras: `plot` (matplotlib), `hme` (gpytorch + CPU torch), `docs`
  (mkdocs-material, mkdocstrings, mkdocs-jupyter), `dev` (pytest, pytest-cov,
  ruff).
- The **`hme` extra** is now `gpytorch` + CPU `torch` instead of `gpflow` +
  `tensorflow`. `sims_pars.fit.hme.emulator.GPREmulator` is reimplemented on
  GPyTorch — an exact GP (`ConstantMean` + `ScaleKernel(RBFKernel)`,
  `GaussianLikelihood`), Adam optimisation, float64. `predict()` returns 1-D
  `(N,)` mean/variance arrays (was gpflow's `(N,1)`). There are **no GPU/device
  options**; the emulator runs on CPU. This removes TensorFlow from the
  dependency tree and restores Python 3.14 support for history matching.
- `sims_pars.prob` no longer exports the private helper types; import
  `PositiveFloat` / `NonNegativeFloat` from `sims_pars.prob` if you need them.

### Security

- Script expressions (`ValueLoci`, `FunctionLoci`, distribution definitions) are
  evaluated through the new `sims_pars.util.safe_eval`, which clears
  `__builtins__` (leaving only `min max abs round sum pow len int float bool`)
  and restricts the AST to an arithmetic / function-call allow-list. Attribute
  access, subscripting, comprehensions, lambdas and imports are rejected — with a
  span in strict mode, not a crash.

### Added

- **`sims_pars.pcore`** — a real front end for PCore scripts.
  - a whitespace-aware lexer (strings keep their spaces; `#` / `//` / `--` line
    comments; identifiers never start with a digit),
  - a recursive-descent parser: `parse(text)` **never raises** and recovers at
    the next line, so one bad statement never stops the parse and nothing is
    silently dropped,
  - located diagnostics — `Diagnostic(span, severity, code, message, hint)` with
    a rendered source frame; unknown-distribution "did you mean", missing-argument
    and cyclic-dependency checks,
  - `compile_script(text)` and `bayes_net_from_script(text, strict=True)` raise
    `DiagnosticError` on any error; a lenient path is kept (see *Deprecated*),
  - a 24-script **compatibility oracle** (`tests/pcore/corpus/`,
    `scripts/build_pcore_corpus.py`) that proves the new front end produces a
    byte-identical `BayesianNetwork` to the legacy parser for every script in
    the repo and the notebooks,
  - a `sims-pars check model.pcore` CLI,
  - a written language specification, `docs/spec/pcore.md`.
- **Documentation site** — MkDocs-Material under `docs/`, `mkdocs.yml`,
  `scripts/stage_tutorials.py`; concept pages, an API reference (mkdocstrings),
  the four tutorial notebooks, and the PCore spec. Deployed to GitHub Pages by
  `.github/workflows/docs.yml`.
- **CI** — `.github/workflows/ci.yml`: `ruff check` + `pytest --cov` on a
  3.10–3.14 matrix, plus a strict docs build.
- pytest test suite (migrated from `unittest`), `tests/conftest.py` with a
  deterministic RNG fixture, and coverage configuration.

### Fixed

- `bayes_net_from_json` raised on a `to_json()` round-trip — it froze the wrong
  object, read a `RVRoots` key that `form_js` never writes, and set
  name-mangled attributes from module scope. It now calls `bn.complete()`.
- `ParameterCore.clone` and `PseudoParameterCore` child-copy referenced the
  removed `LogLikelihood` / `LogPrior` attributes.
- `ApproxBayesCom.sample_posteriors` returned raw JSON dicts instead of
  `Particle`s on the non-parallel path, then crashed in `flatten`.
- `SimulationActor.read_upstream` used `except KeyError or AttributeError or
  TypeError`, which only caught `KeyError`; it now catches all three.
- `SimulationGroup.__repr__` had one more `%`-argument than placeholders.
- `AbsData.ev` used `@lru_cache` on an instance method (a leak); now
  `@cached_property`.
- `Monitor` stacked a new stream/file handler onto the named logger every time
  it was constructed; handler registration is now guarded.
- `sims_pars.fitting.fitter` and `sims_pars.fit.ga.*` had `from fit.…` imports
  missing the `sims_pars.` prefix.

### Performance

- `MathExpression` compiles its expression to a code object once, instead of
  re-`eval`-ing a string on every call.
- `DistributionLoci` caches the built distribution while its parent values are
  unchanged — the dominant cost in fitting loops.
- The fitting objective `Domain` is computed once (`functools.cached_property`),
  not rebuilt on every attribute access.
- `evaluate_nodes` sums lazily instead of allocating an array for a scalar
  reduction.

### Deprecated

- `bayes_net_from_script(script)` without `strict=True` keeps the legacy
  line-by-line parser, which silently skips anything it cannot classify. It stays
  for one release; new code should pass `strict=True` or use
  `sims_pars.pcore.compile_script`.

### Known issues (annotated with `TODO` markers, not fixed here)

- `sims_pars.data.*` — every module imports the predecessor package `epidag`;
  `import sims_pars.data` fails. Dead code, kept pending a decision.
  (`TODO(epidag)`)
- `sims_pars.fitting.*` and `sims_pars.fit.ga.*` — written against a
  pre-"Remove likelihood" `Chromosome` (`LogPrior` / `LogLikelihood` /
  `is_*_evaluated`). They import cleanly but raise at runtime; use
  `sims_pars.fit.*` instead. (`TODO(chromosome-api)`)

### Commits

| | |
|---|---|
| `build:` | consolidate packaging into a single `pyproject.toml` |
| `feat!:` | run on Python 3.14; pydantic 2 / NumPy 2; `safe_eval`; latent-bug fixes; performance |
| `test:`  | migrate to pytest and broaden coverage |
| `docs:`  | add the MkDocs-Material site and CI |
| `docs:`  | mark the runtime-broken modules with `TODO` annotations |
| `feat(pcore):` | real front end for PCore scripts (spec + lexer + parser + diagnostics) |
| `feat(hme):` | replace gpflow/tensorflow with GPyTorch (CPU only) |
