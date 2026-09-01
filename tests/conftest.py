import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed_rng():
    """Deterministic global RNG for every test."""
    np.random.seed(1234)


@pytest.fixture
def pcore_script():
    return '''
    PCore Regression {
        x = 1
        y = x + 1
        z = y + 1
    }
    '''


@pytest.fixture
def rv_script():
    return '''
    PCore BetaBin {
        al = 1
        be = 1
        p1 ~ beta(al, be)
        p2 ~ beta(al, be)
        x1 ~ binom(10, p1)
        x2 ~ binom(20, p2)
    }
    '''


@pytest.fixture
def bn(rv_script):
    from sims_pars import bayes_net_from_script
    return bayes_net_from_script(rv_script)
