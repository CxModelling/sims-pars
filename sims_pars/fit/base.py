import numpy as np
from abc import ABCMeta, abstractmethod
from sims_pars.fn import evaluate_nodes, sample
from sims_pars.monitor import Monitor
from sims_pars.fit.targets import AbsData
from sims_pars.fit.results import ParameterSet
from sims_pars.bayesnet import BayesianNetwork, Chromosome, bayes_net_from_json, bayes_net_from_script
from typing import Union

__author__ = 'Chu-Chang Ku'
__all__ = ['Particle', 'Domain', 'DataModel', 'Fitter']


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

    def to_json(self):
        return {
            'Pars': dict(self.Pars),
            'Sims': dict(self.Sims),
            'Notes': dict(self.Notes)
        }

    @staticmethod
    def from_json(js):
        p = Particle(js['Pars'], js['Sims'])
        p.Notes.update(js['Notes'])
        return p


class DataModel(metaclass=ABCMeta):
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
            res.append(Domain(name=node, tp=d.Type, upper=d.Upper, lower=d.Lower, loc=d.mean(), scale=d.std()))
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

    @abstractmethod
    def simulate(self, pars) -> Particle:
        pass

    def calc_distance(self, particle: Particle) -> float:
        try:
            ys = particle['Ys']
        except KeyError:
            self.flatten(particle)
            ys = particle['Ys']

        di = 0
        for k, dat in self.Data.items():
            di += dat.calc_distance(ys)
        return di

    def calc_likelihood(self, particle: Particle) -> float:
        try:
            ys = particle['Ys']
        except KeyError:
            self.flatten(particle)
            ys = particle['Ys']

        li = 0
        for k, dat in self.Data.items():
            li += dat.calc_likelihood(ys)
        return li

    def flatten(self, particle: Particle) -> None:
        pars = particle.Pars
        particle['Xs'] = [pars[dom.Name] for dom in self.Domain]

        sim = particle.Sims
        particle['Ys'] = {k: sim[k] for k in self.Data.keys()}

    def print(self):
        print('Model: {}'.format(self.BayesianNetwork.Name))
        print('Free parameters: {}'.format(', '.join(self.FreeParameters)))
        print('Exogenous variables: {}'.format(', '.join(self.ExoParameters)))


class IdleModel(DataModel):
    def __init__(self):
        DataModel.__init__(self, {}, bayes_net_from_script('''
                PCore Empty {
        }
        '''))

    def simulate(self, pars) -> Particle:
        return Particle({}, {})


class Fitter(metaclass=ABCMeta):
    def __init__(self, name_logger, **kwargs):
        self.Monitor = Monitor(name_logger)
        self.Settings = dict(self.DefaultSettings)
        self.update_settings(**kwargs)
        self.Collector = ParameterSet()
        self.Model = IdleModel()
        self.State = None

    def set_log_path(self, filename):
        self.Monitor.set_log_path(filename=filename)

    def attach(self, model):
        self.Model = model

    def close(self):
        self.Model = IdleModel()

    def info(self, msg):
        self.Monitor.info(msg)

    def error(self, msg):
        self.Monitor.info(msg)

    @property
    def DefaultSettings(self) -> dict:
        return {
            'parallel': True,
            'n_core': 4,
            'verbose': 5
        }

    def update_settings(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.Settings:
                self.Settings[k] = v

    def fit(self, model: DataModel, **kwargs):
        self.attach(model)
        self.Collector = ParameterSet()
        self.update_settings(**kwargs)
        self.initialise()
        self.update()
        self.terminate()

    @abstractmethod
    def initialise(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def terminate(self):
        pass

    @abstractmethod
    def sample_posteriors(self, n_collect=300) -> ParameterSet:
        pass
