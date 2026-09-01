# Changelog

## 3.0.0 (unreleased)

### Breaking

- **Python 3.10+** required (was 3.9).
- **pydantic 2** required (was pydantic 1).
- **NumPy 2** required.
- Packaging consolidated into a single `pyproject.toml`. `src/setup.py`,
  `requirements/` and `environments/env_dev.yml` were removed; use
  `pip install -e ".[dev]"`.
- The `hme` extra is now `gpytorch` + CPU `torch` instead of `gpflow` +
  `tensorflow`. `sims_pars.fit.hme.emulator.GPREmulator` is reimplemented on
  GPyTorch (exact GP, `ScaleKernel(RBFKernel)`, Adam); `predict()` returns 1-D
  mean/variance arrays. There are no GPU/device options — the emulator runs on
  CPU. This drops the TensorFlow dependency entirely and restores Python 3.14
  support for history matching.

### Security

- Script expressions are now evaluated through `sims_pars.util.safe_eval`, which
  strips builtins and restricts the AST to an arithmetic / function-call
  allow-list. Attribute access, subscripting, comprehensions, lambdas and
  imports are rejected.

### Fixed

- `bayes_net_from_json` no longer raises on a `to_json()` round-trip (froze the
  wrong object, read a missing `RVRoots` key, mangled cached attributes).
- `ParameterCore.clone` / child-copy referenced the removed
  `LogLikelihood` / `LogPrior` attributes.
- `ApproxBayesCom.sample_posteriors` returned raw dicts instead of `Particle`s
  on the non-parallel path.
- `SimulationActor.read_upstream` only caught `KeyError` (the `or` chain was a
  bug); it now catches `KeyError`, `AttributeError` and `TypeError`.
- Removed `np.Inf` / `np.NaN` usage (removed in NumPy 2).

### Performance

- `MathExpression` compiles its expression once instead of re-`eval`ing a string.
- `DistributionLoci` caches the built distribution when its parents are unchanged.
- Fitting objective `Domain` is computed once (`functools.cached_property`).
- `evaluate_nodes` sums lazily instead of allocating an array.

### Added

- pytest suite, coverage config and a CI matrix (3.10–3.14).
- MkDocs-Material documentation site published to GitHub Pages.
- `sims_pars.pcore` — a real front end for PCore scripts: a whitespace-aware
  lexer, a recursive-descent parser and located diagnostics. `parse(text)` never
  raises; `compile_script(text)` / `bayes_net_from_script(text, strict=True)`
  report every problem with a line and column instead of crashing or silently
  dropping the line. Produces a byte-identical network to the legacy parser for
  every script in `tests/pcore/corpus/`. A `sims-pars check` CLI and a written
  language spec (`docs/spec/pcore.md`) come with it.
