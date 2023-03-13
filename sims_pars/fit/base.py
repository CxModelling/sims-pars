import numpy as np

from fit.targets import AbsData
from sims_pars.fn import evaluate_nodes, sample
from sims_pars.bayesnet import BayesianNetwork, Chromosome, bayes_net_from_json, bayes_net_from_script
from typing import Union
from collections import namedtuple

__author__ = 'Chu-Chang Ku'
__all__ = ['Particle', 'Domain', 'DataModel']


class Domain:
    def __init__(self, name, tp='float', lower=-np.inf, upper=np.inf, loc=0, scale=1):
        self.Name = name
        self.Type = tp
        self.Lower, self.Upper = lower, upper
        self.Location, self.Scale = loc, scale

    def __str__(self):
        return f'Domain({self.Name}, {self.Type}, LU=[{self.Lower}, {self.Upper}], LS=[{self.Location}, {self.Scale}]))'

    __repr__ = __str__


class Particle:
    def __init__(self, pars, sims):
        self.Pars = pars
        self.Sims = sims
        self.Notes = dict()

    def __setitem__(self, key, value):
        self.Notes[key] = value

    def __getitem__(self, item):
        return self.Notes[item]


class DataModel:
    def __init__(self, data: dict[str, AbsData], bn: Union[BayesianNetwork, str, dict], exo=None):
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
        self.Data = data

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
            res.append(Domain(Name=node, Type=d.Type, Upper=d.Upper, Lower=d.Lower, Location=d.mean(), Scale=d.std()))
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

    def simulate(self, pars) -> Particle:
        return Particle(pars, pars)

    def flatten(self, particle: Particle) -> None:
        pars = particle.Pars
        xs = [pars[dom.Name] for dom in self.Domain]
        particle['Xs'] = [pars[dom.Name] for dom in self.Domain]

        sim = particle.Sims
        particle['Ys'] = [sim[k] for k in self.Data.items()]

    def print(self):
        print('Model: {}'.format(self.BayesianNetwork.Name))
        print('Free parameters: {}'.format(', '.join(self.FreeParameters)))
        print('Exogenous variables: {}'.format(', '.join(self.ExoParameters)))


class AbsFitter:
    pass

