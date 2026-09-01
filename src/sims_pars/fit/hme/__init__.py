# TODO(hme-extra): requires the optional `hme` extra (gpflow/tensorflow);
# `sims_pars.fit.hme.emulator` fails to import without it. Also unverified
# against the current Chromosome / DataModel API.
from sims_pars.fit.hme.alg import BayesHistoryMatching
