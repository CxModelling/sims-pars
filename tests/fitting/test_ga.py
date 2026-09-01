"""Genetic-algorithm point estimation on the BetaBin toy."""
import numpy as np
import pytest

from sims_pars.fit import GeneticAlg
from sims_pars.fit.ga.cross import get_crossover
from sims_pars.fit.ga.mutate import get_mutator
from sims_pars.fit.ga.select import get_selector
from sims_pars.fit.ga.util import sample_prior, fitness_of
from sims_pars.fit.toys import get_betabin


@pytest.fixture
def model():
    return get_betabin((4, 12))


def test_operators_are_registered_and_shape_preserving(model):
    keys = model.FreeParameters
    g1 = {k: model.sample_prior()[k] for k in keys}
    g2 = {k: model.sample_prior()[k] for k in keys}

    for name in ('average', 'shuffle'):
        c1, c2 = get_crossover(name).crossover(g1, g2, keys)
        assert set(c1) == set(keys) and set(c2) == set(keys)

    pool = [{k: model.sample_prior()[k] for k in keys} for _ in range(20)]
    mut = get_mutator('rw')
    mut.set_scales(pool, keys)
    assert set(mut.Scales) == set(keys)
    assert set(mut.propose(pool[0], keys)) == set(keys)


def test_selectors_return_requested_count(model):
    keys = model.FreeParameters
    pop = [sample_prior(model)[0] for _ in range(30)]
    scored = [(fitness_of(pt, 'MAP'), {k: pt.Pars[k] for k in keys}) for pt in pop]

    assert len(get_selector('tour(4)').select(scored, 25)) == 25
    assert len(get_selector('importance').select(scored, 25)) == 25


@pytest.mark.parametrize('parallel', [False, True])
def test_genetic_alg_recovers_the_betabin_point(model, parallel):
    np.random.seed(1234)
    alg = GeneticAlg(n_collect=80, max_round=20, parallel=parallel, verbose=0)
    alg.fit(model)

    post = alg.sample_posteriors(80)
    df = post.to_df()
    assert len(df) == 80

    # BetaBin observed (4/10, 12/20) -> (0.4, 0.6). The toy distance is discrete
    # and flat at the optimum, so the population concentrates only loosely.
    assert abs(df['p1'].mean() - 0.4) < 0.2
    assert abs(df['p2'].mean() - 0.6) < 0.2
    assert np.isfinite(alg.State.MaxFitness)
    assert 'Best' in post.Notes
