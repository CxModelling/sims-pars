from sims_pars.fn import evaluate_nodes, sample
from sims_pars.bayesnet import BayesianNetwork, Chromosome, bayes_net_from_json, bayes_net_from_script
from abc import ABCMeta, abstractmethod
from typing import Union
from collections import namedtuple

__author__ = 'Chu-Chang Ku'
__all__ = []


Domain = namedtuple('Domain', ('Name', 'Type', 'Lower', 'Upper'))


class Particle:
    def __init__(self, xs, pars, sims):
        self.Xs = xs
        self.Pars = pars
        self.Sims = sims
        self.Notes = dict()

    def __setitem__(self, key, value):
        self.Notes[key] = value


class Simulator(metaclass=ABCMeta):
    def __init__(self,  bn: Union[BayesianNetwork, str, dict], exo=None):
        self.ExoParameters = dict(exo) if exo is not None else dict()
        self.FreeParameters = dict()

        if isinstance(bn, str):
            bn = bayes_net_from_script(bn)
        elif isinstance(bn, dict):
            bn = bayes_net_from_json(bn)

        self.FreeParameters = [node for node in bn.Order if bn.is_rv(node) and node not in self.ExoParameters]
        self.BayesianNetwork = bn

        # Exclude non-float
        # todo
        p0 = self.sample_prior()
        pfs = [k for k, v in dict(p0).items() if isinstance(v, float)]
        self.FreeParameters = [node for node in self.FreeParameters if node in pfs]

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
            res.append(Domain(Name=node, Type=d.Type, Upper=d.Upper, Lower=d.Lower))
        return res

    def serve_from_json(self, js: dict):
        p = self.serve(js['Locus'])
        p.LogPrior, p.LogLikelihood = js['LogPrior'], js['LogLikelihood']
        return p

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

    def evaluate(self, pars: Chromosome) -> float:
        # prevent re-evaluation
        if not pars.is_likelihood_evaluated():
            pars.LogLikelihood = self.calc_likelihood(pars)
        return pars.LogLikelihood

    @abstractmethod
    def simulate(self, pars):
        pass

    @abstractmethod
    def link_likelihood(self, sim):
        pass

    def print(self):
        print('Model: {}'.format(self.BayesianNetwork.Name))
        print('Free parameters: {}'.format(', '.join(self.FreeParameters)))
        print('Exogenous variables: {}'.format(', '.join(self.ExoParameters)))
