# Distributions

Distributions are named in scripts by a short tag and built through a registry
(`DistributionCentre`). Parse one directly
with `parse_distribution`.

```python
from sims_pars.prob import parse_distribution

d = parse_distribution('gamma(0.1, 0.1)')
d.mean(), d.std(), d.Interval
d.sample(5)
```

## Registered tags

| Tag | Parameters | Backed by |
|-----|------------|-----------|
| `k` | `k` | constant |
| `unif` | `min`, `max` | `scipy.stats.uniform` |
| `norm` | `mean`, `sd` | `scipy.stats.norm` |
| `lnorm` | `meanlog`, `sdlog` | `scipy.stats.lognorm` |
| `gamma` | `shape`, `rate` | `scipy.stats.gamma` |
| `invgamma` | `a`, `rate` | `scipy.stats.invgamma` |
| `exp` | `rate` | `scipy.stats.expon` |
| `beta` | `shape1`, `shape2` | `scipy.stats.beta` |
| `chisq` | `df` | `scipy.stats.chi2` |
| `triangle` | `a`, `m`, `b` | `scipy.stats.triang` |
| `binom` | `size`, `prob` | `scipy.stats.binom` |
| `pois` | `lam` | `scipy.stats.poisson` |
| `cat` | `kv` (dict) | categorical |

Parameters may be given positionally or by name; unspecified ones fall back to
the creator's default. Parameter constraints (e.g. a positive rate) are enforced
by pydantic and raise `ValidationError`.

## Custom maths functions

```python
from sims_pars import add_math_func

add_math_func('sigmoid', lambda x: 1 / (1 + 2.718281828 ** -x))
```
