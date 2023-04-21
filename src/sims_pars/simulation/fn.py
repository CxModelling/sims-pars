import sims_pars as dag
from sims_pars.simulation.nodeset import NodeSet
from sims_pars.simulation.simucore import SimulationCore
__author__ = 'TimeWz667'
__all__ = [
        'as_simulation_core',
        'get_all_fixed_sc',
        'get_all_float_sc',
        'find_free_parameters'
    ]


def as_simulation_core(bn, ns: NodeSet = None):
    """
    a blueprint of a simulation model based on given a Bayesian network.
    It describes every node in the network as 1) fixed variable, 2) random variable, 3) exogenous distribution
    :param bn: epidag.BayesNet, a Bayesian Network
    :param ns: name of root group
    :return: a simulation model
    """
    if not ns:
        ns = NodeSet('Root', as_floating=bn.DAG.leaves())
    ns.inject_bn(bn)
    return SimulationCore(bn, ns)


def get_all_float_sc(script):
    bn = dag.bayes_net_from_script(script)

    ns = NodeSet('Root', as_floating=bn.Order)
    sm = as_simulation_core(bn, ns)
    return sm


def get_all_fixed_sc(script):
    bn = dag.bayes_net_from_script(script)

    ns = NodeSet('Root', as_fixed=bn.Order)
    sm = as_simulation_core(bn, ns)
    return sm


def find_free_parameters(sm, exo=None):
    p = sm.generate(exo=exo)
    bn = sm.BN

    return [k for k in p.keys() if k in bn.RVRoots]


if __name__ == '__main__':
    scr = '''
    PCore ABC {
        pa0 = 0.4
        A ~ binom(10, pa0)
        B ~ norm(0, 1)
        C = A + B
    }  
    '''

    sc1 = get_all_float_sc(scr)
    sc1.deep_print()

    p1 = sc1.generate('p1')
    print(p1)
    print(p1.list_actors())

    print(find_free_parameters(sc1))

    sc2 = get_all_fixed_sc(scr)
    sc2.deep_print()

    p2 = sc2.generate('p2')
    print(p2)
    print(p2.list_actors())

    print(find_free_parameters(sc2))
