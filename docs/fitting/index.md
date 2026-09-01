# Fitting

`sims_pars.fit` calibrates a model to data with approximate Bayesian
computation. You describe the model by subclassing
[`DataModel`][sims_pars.fit.base.DataModel] and implementing `simulate`.

```python
from sims_pars.fit.toys import get_betabin
from sims_pars.fit import ApproxBayesComSMC

model = get_betabin((7, 14))

alg = ApproxBayesComSMC(parallel=False, n_iter=1000, max_round=10)
alg.fit(model)
post = alg.sample_posteriors(500)
post.to_df().describe()
```

## Available algorithms

| Class | Module | Idea |
|-------|--------|------|
| [`ApproxBayesCom`][sims_pars.fit.abcom.alg.ApproxBayesCom] | `fit.abcom` | fixed-epsilon rejection ABC |
| [`ApproxBayesComSMC`][sims_pars.fit.abc_smc.alg.ApproxBayesComSMC] | `fit.abc_smc` | sequential Monte Carlo ABC with adaptive epsilon |

Both accept `parallel=True` (joblib) or `parallel=False`. `ApproxBayesComSMC`
keeps its `State` between `update()` calls and can be seeded from a previous
run's particles via `set_parents(...)`.

!!! note "Status"
    `sims_pars.fit.hme` (history matching) needs the `hme` extra
    (gpytorch + CPU torch). The `sims_pars.fitting` package and `sims_pars.fit.ga`
    are mid-refactor and not part of the supported surface yet.
