# Simulation core

A [`SimulationCore`][sims_pars.simulation.simucore.SimulationCore] wraps a
Bayesian network together with a [`NodeSet`][sims_pars.simulation.nodeset.NodeSet]
that decides, per node, whether it is **fixed** (drawn once and shared) or
**floating** (redrawn for each actor). Build one with
[`as_simulation_core`][sims_pars.simulation.fn.as_simulation_core].

```python
from sims_pars.simulation.fn import get_all_fixed_sc, find_free_parameters

sc = get_all_fixed_sc('''
PCore ABC {
    pa0 = 0.4
    A ~ binom(10, pa0)
    B ~ norm(0, 1)
    C = A + B
}
''')

p = sc.generate('run-1')
p['C'] == p['A'] + p['B']          # True
find_free_parameters(sc)           # ['A', 'B'] — the RV roots
```

## Fixed vs floating

- `get_all_fixed_sc(script)` — every node materialised in the parameter core.
- `get_all_float_sc(script)` — leaf nodes become *actors* sampled on demand
  (`p.list_actors()`), so a single core can spawn many stochastic realisations.

## Hierarchies

`NodeSet` objects nest via `new_child(...)`, letting a parent core `breed`
child cores that inherit fixed values but resample floating ones — useful for
individual-level or group-level model structure.
