"""End-to-end smoke tests for the ABC fitters on the toy beta-binomial model."""
import numpy as np
import pytest

from sims_pars.fit import ApproxBayesCom, ApproxBayesComSMC
from sims_pars.fit.toys import get_betabin


@pytest.fixture
def model():
    # data (x1, x2) = (7, 14) with n = (10, 20)  ->  p1 ~ 0.7, p2 ~ 0.7
    return get_betabin((7, 14))


def test_approx_bayes_com_runs(model):
    alg = ApproxBayesCom(parallel=False, n_test=120, p_test=0.1)
    alg.fit(model)
    post = alg.sample_posteriors(120)
    df = post.to_df()
    assert len(df) == 120
    assert 0.4 < df['p1'].mean() < 0.95


def test_approx_bayes_com_smc_runs(model):
    alg = ApproxBayesComSMC(parallel=False, n_iter=120, max_round=3)
    alg.fit(model)
    post = alg.sample_posteriors(100)
    df = post.to_df()
    assert len(df) > 0
    assert np.isfinite(df['p1'].mean())


def test_smc_state_is_resumable(model):
    alg = ApproxBayesComSMC(parallel=False, n_iter=100, max_round=2)
    alg.fit(model)
    assert alg.State is not None
    round_1 = alg.State.Round
    alg.update()
    assert alg.State.Round >= round_1
