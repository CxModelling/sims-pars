# sims-pars

**Serving stochastic parameters to simulation models.**

`sims-pars` turns a compact text description of a probabilistic model — a
*PCore script* — into a Bayesian network you can sample from, condition on data,
and fit with approximate Bayesian computation.

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
# {'beta': 12.3, 'gamma': 0.44, 'r0': 27.9}
```

## What's here

| Area | Module | Notes |
|------|--------|-------|
| Model description | [`sims_pars.bayesnet`](api/bayesnet.md) | PCore script parser, DAG, loci types |
| Distributions | [`sims_pars.prob`](api/prob.md) | scipy-backed distribution registry |
| Simulation | [`sims_pars.simulation`](api/simulation.md) | hierarchical parameter cores |
| Fitting | [`sims_pars.fit`](fitting/index.md) | ABC and ABC-SMC |

## Install

```bash
pip install sims-pars
```

See [Installation](installation.md) for optional extras and development setup.

## Requirements

Python 3.10+ · NumPy 2 · SciPy · pandas · pydantic 2 · networkx
