# ABC & ABC-SMC

::: sims_pars.fit.abcom.alg.ApproxBayesCom
    options:
      show_root_heading: true
      members: [initialise, sample_posteriors]

::: sims_pars.fit.abc_smc.alg.ApproxBayesComSMC
    options:
      show_root_heading: true
      members: [initialise, update, sample_posteriors, set_parents]

## Defining a model

::: sims_pars.fit.base.DataModel

::: sims_pars.fit.base.Particle
