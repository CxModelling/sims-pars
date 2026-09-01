# sims-pars

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

<https://cxmodelling.github.io/sims-pars/>

## Development

```bash
pip install -e ".[dev,plot]"
pytest
ruff check src tests
```
