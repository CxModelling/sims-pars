from abc import ABCMeta, abstractmethod
from sims_pars.monitor import Monitor
from sims_pars.fitting.target import AbsTarget
from sims_pars.fitting.results import ParameterSet
from joblib import Parallel, delayed
import numpy as np

__author__ = 'TimeWz667'
__all__ = ['Fitter']


class Fitter(metaclass=ABCMeta):
    def __init__(self, name_logger, **kwargs):
        self.Monitor = Monitor(name_logger)
        self.Parameters = dict(kwargs)
        self.__uses_pseudo_likelihood = True
        self.Settings = dict(self.DefaultSettings)

        for k, v in kwargs.items():
            if k in self.Settings:
                self.Settings[k] = v

    def set_log_path(self, filename):
        self.Monitor.set_log_path(filename=filename)

    def info(self, msg):
        self.Monitor.info(msg)

    def error(self, msg):
        self.Monitor.info(msg)

    @property
    def DefaultSettings(self) -> dict:
        return {}

    @abstractmethod
    def fit(self, model: AbsTarget) -> ParameterSet:
        pass

    def uses_pseudo_likelihood(self):
        return self.__uses_pseudo_likelihood

    def update(self, model: AbsTarget, res, **kwargs) -> ParameterSet:
        raise AttributeError("The posterior is not updatable with this algorithm")


class PriorSampling(Fitter):
    def __init__(self, **kwargs):
        Fitter.__init__(self, "Prior", **kwargs)

    @property
    def DefaultSettings(self) -> dict:
        return {
            'n_sim': 300,
            'parallel': False,
            'n_core': 4,
            'verbose': 5
        }

    def fit(self, model: AbsTarget) -> ParameterSet:
        self.info('Initialise condition')

        n_sim = self.Settings['n_sim']

        res = ParameterSet('Prior')

        def draw(m):
            p = m.sample_prior()
            li = m.evaluate(p)
            i = 1
            while np.isinf(li):
                p = m.sample_prior()
                li = m.evaluate(p)
                i += 1
                if i > 20:
                    break
            return p.to_json()

        if self.Settings['parallel']:
            self.info('Start a parallel sampler')
            with Parallel(n_jobs=self.Settings['n_core'], verbose=self.Settings['verbose']) as parallel:
                samples = parallel(delayed(draw)(model) for _ in range(n_sim))
            for sample in samples:
                p = model.SimulationCore.generate(exo=sample['Locus'])
                p.LogLikelihood = sample['LogLikelihood']
                res.append(p)

        else:
            self.info('Start a sampler')

            for _ in range(n_sim):
                p = draw(model)
                res.append(model.SimulationCore.generate(exo=p))

        res.finish()
        self.info('Finish')
        return res


if __name__ == '__main__':
    from sims_pars.simulation import get_all_fixed_sc
    import logging

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

    model0 = BetaBin(sc0, exo={'n2': 20})

    alg = PriorSampling(parallel=True)
    alg.Monitor.add_handler(logging.StreamHandler())

    res0 = alg.fit(model0)
    print(res0.DF.describe())
