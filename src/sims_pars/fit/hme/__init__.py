# Requires the optional `hme` extra (gpytorch / torch, CPU-only);
# `sims_pars.fit.hme.emulator` fails to import without it.
from sims_pars.fit.hme.alg import BayesHistoryMatching
