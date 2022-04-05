from sims_pars.fitting.base import AbsObjective
from abc import ABCMeta, abstractmethod
from sims_pars.fn import evaluate_nodes, sample
from sims_pars.bayesnet import BayesianNetwork, Chromosome


__author__ = 'Chu-Chang Ku'
__all__ = ['AbsObjectiveSimBased']


class AbsObjectiveSimBased(AbsObjective, metaclass=ABCMeta):
    def __init__(self, bn: BayesianNetwork, exo=None):
        AbsObjective.__init__(self, exo)
        self.FreeParameters = list(bn.RVRoots)
        self.BayesianNetwork = bn

    def serve(self, p: dict):
        p = dict(p)
        p.update(self.ExoParameters)
        pars = Chromosome(sample(self.BayesianNetwork, p))
        self.evaluate_prior(pars)
        return pars

    @property
    def Domain(self):
        p = self.sample_prior()
        res = []
        for node in self.FreeParameters:
            d = self.BayesianNetwork[node].get_distribution(p)
            res.append({'Name': node, 'Type': d.Type, 'Upper': d.Upper, 'Lower': d.Lower})
        return res

    def sample_prior(self):
        pars = sample(self.BayesianNetwork, self.ExoParameters)
        pars.update(self.ExoParameters)
        pars = Chromosome(pars)
        self.evaluate_prior(pars)
        return pars

    def evaluate_prior(self, p: Chromosome):
        if not p.is_prior_evaluated():
            p.LogPrior = evaluate_nodes(self.BayesianNetwork, p)
        return p.LogPrior

    def calc_likelihood(self, pars: Chromosome):
        sim = self.simulate(pars)
        if sim is None:
            raise ValueError('Invalid simulation run')
        return self.link_likelihood(sim)

    @abstractmethod
    def simulate(self, pars):
        pass

    @abstractmethod
    def link_likelihood(self, sim):
        pass

    def print(self):
        print('Model: {}'.format(self.BayesianNetwork.Name))
        AbsObjective.print(self)
