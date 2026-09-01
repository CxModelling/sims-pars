import numpy as np

from sims_pars import bayes_net_from_script
from sims_pars.simulation.fn import (
    as_simulation_core,
    find_free_parameters,
    get_all_fixed_sc,
    get_all_float_sc,
)

SCRIPT = '''
PCore ABC {
    pa0 = 0.4
    A ~ binom(10, pa0)
    B ~ norm(0, 1)
    C = A + B
}
'''


def test_all_fixed_core_materialises_every_node():
    sc = get_all_fixed_sc(SCRIPT)
    p = sc.generate('p1')
    assert p['pa0'] == 0.4
    assert p['C'] == p['A'] + p['B']


def test_free_parameters_are_the_rv_roots():
    fixed = get_all_fixed_sc(SCRIPT)
    assert set(find_free_parameters(fixed)) == {'A', 'B'}


def test_all_float_core_defers_nodes_to_actors():
    sc = get_all_float_sc(SCRIPT)
    p = sc.generate('p1')
    assert set(p.list_actors()) == {'A', 'B', 'C'}


def test_generate_is_deterministic_under_seed():
    sc = get_all_fixed_sc(SCRIPT)
    np.random.seed(7)
    a = sc.generate('a')['B']
    np.random.seed(7)
    b = sc.generate('b')['B']
    assert a == b


def test_as_simulation_core_accepts_explicit_bn():
    bn = bayes_net_from_script(SCRIPT)
    sc = as_simulation_core(bn)
    assert sc.BN is bn
