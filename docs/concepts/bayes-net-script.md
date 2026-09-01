# The PCore script

A *PCore script* is a small text block that describes a Bayesian network. Parse
it with [`bayes_net_from_script`][sims_pars.bayesnet.bn.bayes_net_from_script].

```text
PCore <Name> {
    <definition>
    <definition>
    ...
}
```

## Node (loci) types

Each line inside the braces defines one node. The syntax picks the type:

| Line | Loci type | Meaning |
|------|-----------|---------|
| `x = 5` | `ValueLoci` | a constant |
| `x = exp(3 * k)` | `FunctionLoci` | deterministic function of other nodes |
| `x ~ norm(mu, 1)` | `DistributionLoci` | a random variable |
| `x = f(y, z)` | `PseudoLoci` | placeholder that is filled in elsewhere |
| `x` | `ExoValueLoci` | exogenous — must be supplied at sample time |
| `x ~ binom(5, p)  # a note` | — | trailing `#` adds a description |

Names that appear in an expression but are never defined become exogenous nodes
automatically.

## Expressions

Expressions are evaluated in a restricted sandbox (see
[`safe_eval`][sims_pars.util.safe_eval]): arithmetic, comparisons, the functions
in `MATH_FUNC` (`exp`, `log`, `sin`, `logit`,
`ifelse`, `step`, …) and a handful of safe builtins (`min`, `max`, `abs`, …).
Attribute access, subscripting, comprehensions and imports are rejected.

```python
from sims_pars import bayes_net_from_script, sample

bn = bayes_net_from_script('''
PCore Growth {
    r ~ norm(0.03, 0.01)
    k = 1000
    n0 = 10
    n1 = n0 * exp(r)
}
''')

sample(bn, {'n0': 5})
```

## Round-tripping

`bn.to_json()` / `bayes_net_from_json`, and `bn.to_script()` /
`bayes_net_from_script`, both reproduce an equivalent network.

## Strict parsing

`bayes_net_from_script(script, strict=True)` parses through
[`sims_pars.pcore`](../spec/pcore.md): every problem is reported with a line and
column instead of crashing or being silently skipped. `sims-pars check
model.pcore` runs the same checks from the command line. See the
[language spec](../spec/pcore.md) for the grammar and the diagnostic codes.
