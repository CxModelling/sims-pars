from abc import ABCMeta, abstractmethod
from sims_pars.fn import evaluate_nodes, sample
from sims_pars.bayesnet import BayesianNetwork, Chromosome, bayes_net_from_json, bayes_net_from_script
from typing import Union
from collections import namedtuple

__author__ = 'Chu-Chang Ku'
__all__ = []


class Particle:
    def __init__(self, xs, pars, sims):
        self.Xs = xs
        self.Pars = pars
        self.Sims = sims
        self.Notes = dict()

    def __setitem__(self, key, value):
        self.Notes[key] = value



class AbsObjective(metaclass=ABCMeta):
    def __init__(self, exo=None):
        self.ExoParameters = dict(exo) if exo is not None else dict()
        self.FreeParameters = dict()

    @property
    @abstractmethod
    def Domain(self):
        pass

    @abstractmethod
    def serve(self, p: dict):
        raise AttributeError('Unknown parameter definition')

    def serve_from_json(self, js: dict):
        p = self.serve(js['Locus'])
        p.LogPrior, p.LogLikelihood = js['LogPrior'], js['LogLikelihood']
        return p

    @abstractmethod
    def sample_prior(self):
        pass

    @abstractmethod
    def evaluate_prior(self, pars: Chromosome):
        pass

    @abstractmethod
    def calc_likelihood(self, pars: Chromosome):
        pass

    def evaluate(self, pars: Chromosome) -> float:
        # prevent re-evaluation
        if not pars.is_likelihood_evaluated():
            pars.LogLikelihood = self.calc_likelihood(pars)
        return pars.LogLikelihood

    def print(self):
        print('Free parameters: {}'.format(', '.join(self.FreeParameters)))
        print('Exogenous variables: {}'.format(', '.join(self.ExoParameters)))


class AbsObjectiveBN(AbsObjective, metaclass=ABCMeta):
    def __init__(self, bn: Union[BayesianNetwork, str, dict], exo=None):
        AbsObjective.__init__(self, exo)
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

    @abstractmethod
    def calc_likelihood(self, pars: Chromosome):
        pass

    def print(self):
        print('Model: {}'.format(self.BayesianNetwork.Name))
        AbsObjective.print(self)


class AbsObjectiveSimBased(AbsObjectiveBN, metaclass=ABCMeta):
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

