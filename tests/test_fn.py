import numpy as np

from sims_pars import bayes_net_from_script
from sims_pars.fn import evaluate_nodes, sample, sample_chromosome, sample_minimally

SCRIPT = '''
PCore Chain {
    a = 2
    b ~ norm(a, 1)
    c = b + 1
    d ~ norm(c, 1)
}
'''


def _bn():
    return bayes_net_from_script(SCRIPT)


def test_sample_full():
    p = sample(_bn())
    assert set(p) == {'a', 'b', 'c', 'd'}
    assert p['a'] == 2
    assert p['c'] == p['b'] + 1


def test_sample_minimally_sources_toggle():
    bn = _bn()
    sinks, med = sample_minimally(bn, included=['c'], sources=True)
    assert set(sinks) == {'c'}
    assert 'd' not in med
    sinks_only = sample_minimally(bn, included=['c'], sources=False)
    assert set(sinks_only) == {'c'}


def test_evaluate_nodes_finite_for_full_sample():
    bn = _bn()
    p = sample(bn)
    assert np.isfinite(evaluate_nodes(bn, p))


def test_evaluate_nodes_nan_becomes_neg_inf():
    bn = _bn()
    p = sample(bn)
    p['b'] = float('nan')
    assert evaluate_nodes(bn, p) == -np.inf


def test_sample_chromosome_is_evaluated():
    ch = sample_chromosome(_bn())
    assert ch.is_evaluated()
    assert np.isfinite(ch.LogProb)
