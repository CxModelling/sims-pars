"""Round-trip / IO tests for BayesianNetwork (regression guard for the
previously broken ``bayes_net_from_json``)."""
import pytest

from sims_pars.bayesnet import bayes_net_from_json, bayes_net_from_script

SCRIPT = '''
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
def bn():
    return bayes_net_from_script(SCRIPT)


def test_json_round_trip(bn):
    bn2 = bayes_net_from_json(bn.to_json())
    assert bn2.Name == bn.Name
    assert bn2.Order == bn.Order
    assert bn2.is_frozen()
    assert set(bn2.RVRoots) == set(bn.RVRoots)
    assert bn2.Exo == bn.Exo


def test_json_round_trip_is_stable(bn):
    js1 = bn.to_json()
    js2 = bayes_net_from_json(js1).to_json()
    assert js1['Nodes'] == js2['Nodes']
    assert js1['Order'] == js2['Order']


def test_clone_matches_source(bn):
    clone = bn.clone()
    assert clone.Order == bn.Order
    assert clone.to_json()['Nodes'] == bn.to_json()['Nodes']


def test_script_round_trip(bn):
    bn2 = bayes_net_from_script(bn.to_script())
    assert bn2.Order == bn.Order


def test_cyclic_definition_rejected():
    with pytest.raises(AttributeError):
        bayes_net_from_script('''
        PCore Cyclic {
            a = b + 1
            b = a + 1
        }
        ''')
