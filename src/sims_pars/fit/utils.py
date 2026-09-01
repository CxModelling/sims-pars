import numpy as np
from sims_pars.fit import DataModel


__all__ = ['sample_fin']

def sample_fin(model: DataModel, unpack=False):
    di = np.inf
    n_eval = 0
    while np.isinf(di):
        n_eval += 1
        p = model.sample_prior()
        sim = model.simulate(p)
        di = model.calc_distance(sim)

    sim = sim.to_json() if unpack else sim
    return di, sim, n_eval
