# Getting started

A three-chapter walkthrough of the whole `sims-pars` loop, built around one
running example (a small SIR epidemic model). Each chapter is a runnable
notebook; the source lives in
[`notebooks/`](https://github.com/CxModelling/sims-pars/tree/main/notebooks).

1. **[Writing a model](GettingStarted01_The PCore language.ipynb)** — the PCore
   language from scratch: value, function, distribution and exogenous nodes,
   diagnostics, and the v2 additions (type annotations, plates, `include`).
2. **[Sampling & intervention](GettingStarted02_Sampling and intervention.ipynb)**
   — drawing parameter sets, sampling only what you need, and running
   interventions (the do-operator) with `Chromosome.impulse`.
3. **[Fitting a model](GettingStarted03_Fitting a model.ipynb)** — wrapping a
   simulator in a `DataModel` and calibrating it to data with ABC and ABC-SMC,
   plus pointers to the genetic algorithm and history matching.

Once you have the shape of it, the [Tutorials](../tutorials/index.md) go deeper
into each piece and the [API reference](../api/index.md) documents every class.
