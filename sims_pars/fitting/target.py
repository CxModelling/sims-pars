from abc import ABCMeta, abstractmethod
import numpy as np
from sims_pars.fn import evaluate_nodes
from sims_pars.simulation.fn import find_free_parameters

__all__ = ['AbsTarget']


class AbsTarget(metaclass=ABCMeta):
    def __init__(self, sm, exo=None):
        self.ExoParameters = exo
        self.FreeParameters = find_free_parameters(sm, exo)
        self.SimulationCore = sm

    @property
    def Domain(self):
        p = self.sample_prior()

        res = []
        for node in self.FreeParameters:
            d = self.SimulationCore.BN[node].get_distribution(p)
            res.append({'Name': node, 'Type': d.Type, 'Upper': d.Upper, 'Lower': d.Lower})
        return res

    def sample_prior(self):
        return self.SimulationCore.generate(exo=self.ExoParameters)

    def evaluate_prior(self, p):
        if not p.is_prior_evaluated():
            p.LogPrior = evaluate_nodes(self.SimulationCore.BN, p)
        return p.LogPrior

    @abstractmethod
    def calc_likelihood(self, pars):
        pass

    def evaluate(self, pars) -> float:
        # prevent re-evaluation
        if not pars.is_likelihood_evaluated():
            pars.LogLikelihood = self.calc_likelihood(pars)
        return pars.LogLikelihood

    def print(self):
        print('Model: {}'.format(self.SimulationCore.Name))
        print('Free parameters: {}'.format(', '.join(self.FreeParameters)))
        print('Exogenous variables: {}'.format(', '.join(self.ExoParameters)))


if __name__ == '__main__':
    from sims_pars.simulation import get_all_fixed_sc

    class BetaBin(AbsTarget):
        def calc_likelihood(self, pars):
            return -((pars['x1'] - 5) ** 2 + (pars['x2'] - 10) ** 2)

    scr = '''
    PCore BetaBin {
        al = 1
        be = 1
        
        p1 ~ beta(al, be)
        p2 ~ beta(al, be)
        
        x1 ~ binom(10, p1)
        x2 ~ binom(n2, p2) 
    }
    '''

    sc0 = get_all_fixed_sc(scr)

    sc0.deep_print()

    p0 = sc0.generate(exo={'n2': 20})
    print(p0)

    model0 = BetaBin(sc0, exo={'n2': 20})
    model0.print()
    p1 = model0.sample_prior()
    print("Parameters: ", p1)
    print("Likelihood: ", model0.evaluate(p1))
