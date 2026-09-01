import numpy as np
from pydantic import ValidationError
from sims_pars.fit.base import DataModel, Particle

__author__ = 'Chu-Chang Ku'
__all__ = ['fitness_of', 'sample_prior', 'evaluate_gene']

# GA evaluation helpers, written against the current DataModel / Particle API
# (mirrors sims_pars.fit.abc_smc.alg: simulate -> Particle -> calc_distance,
#  parallel workers exchange plain JSON via Particle.to_json / from_json).


def fitness_of(pt: Particle, target: str = 'MAP') -> float:
    """
    Scalar fitness to be *maximised*.

    :param pt: an evaluated particle carrying a ``distance`` note
    :param target: ``'MAP'`` adds the prior log density, anything else is the
        plain goodness-of-fit (``-distance``)
    """
    if pt is None:
        return -np.inf
    d = pt.Notes.get('distance', np.inf)
    if not np.isfinite(d):
        return -np.inf
    if target == 'MAP':
        lp = pt.Pars.LogProb
        return (lp if np.isfinite(lp) else -np.inf) - d
    return -d


def sample_prior(model: DataModel, unpack=False):
    """Draw one prior particle with a finite distance."""
    di = np.inf
    n_eval = 0
    while np.isinf(di):
        n_eval += 1
        p = model.sample_prior()
        sim = model.simulate(p)
        di = model.calc_distance(sim)

    sim = sim.to_json() if unpack else sim
    return sim, n_eval


def evaluate_gene(model: DataModel, gene: dict, unpack=False):
    """
    Serve a free-parameter dict through the model and score it.

    Returns ``(particle_or_None, n_eval)``; ``None`` when the proposal falls
    outside the prior support.
    """
    try:
        p = model.serve(gene)
    except (ValueError, ArithmeticError, ValidationError):
        # proposal drove a downstream distribution parameter out of its domain
        return None, 1
    if np.isinf(p.LogProb):
        return None, 1

    sim = model.simulate(p)
    model.calc_distance(sim)

    sim = sim.to_json() if unpack else sim
    return sim, 1
