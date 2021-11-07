from abc import ABCMeta, abstractmethod
from sims_pars.bayesnet import Chromosome
from sims_pars.monitor import Monitor
from sims_pars.fitting.base import AbsObjective
from sims_pars.fitting.util import draw
from sims_pars.fitting.results import ParameterSet
from joblib import Parallel, delayed


__author__ = 'TimeWz667'
__all__ = ['Fitter', 'PriorSampling', 'ParameterSet']


class IdleModel(AbsObjective):
    def __init__(self):
        AbsObjective.__init__(self)
        self.Parameters = Chromosome({})

    @property
    def Domain(self):
        return []

    def serve(self, p: dict):
        return self.Parameters

    def sample_prior(self):
        return self.Parameters

    def evaluate_prior(self, pars: Chromosome):
        return 0

    def calc_likelihood(self, pars: Chromosome):
        return 0


Idle = IdleModel()


class Fitter(metaclass=ABCMeta):
    def __init__(self, name_logger, **kwargs):
        self.Monitor = Monitor(name_logger)
        self.Settings = dict(self.DefaultSettings)
        self.update_settings(**kwargs)
        self.Collector = ParameterSet()
        self.Model = Idle

    def set_log_path(self, filename):
        self.Monitor.set_log_path(filename=filename)

    def attach(self, model):
        self.Model = model

    def close(self):
        self.Model = Idle

    def info(self, msg):
        self.Monitor.info(msg)

    def error(self, msg):
        self.Monitor.info(msg)

    @property
    def DefaultSettings(self) -> dict:
        return {
            'n_collect': 1000,
            'parallel': True,
            'n_core': 4,
            'verbose': 5
        }

    def update_settings(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.Settings:
                self.Settings[k] = v

    def fit(self, model: AbsObjective, **kwargs):
        self.attach(model)
        self.Collector = ParameterSet()
        self.update_settings(**kwargs)
        self.initialise()
        self.update()
        self.collect()

    @abstractmethod
    def initialise(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def collect(self):
        pass


class PriorSampling(Fitter):
    def __init__(self, **kwargs):
        Fitter.__init__(self, 'Prior', **kwargs)

    def initialise(self):
        self.info('Initialise')

    def update(self):
        pass

    def collect(self):
        n_sim = self.Settings['n_collect']
        self.Collector = res = ParameterSet('Prior')

        if self.Settings['parallel']:
            self.info('Start a parallel sampler')
            with Parallel(n_jobs=self.Settings['n_core'], verbose=self.Settings['verbose']) as parallel:
                samples = parallel(delayed(draw)(self.Model, unpack=True) for _ in range(n_sim))

            for p, _ in samples:
                p = self.Model.serve_from_json(p)
                res.append(p)
            res.keep('N_Eval', sum(i for _, i in samples))

        else:
            self.info('Start a sampler')

            n_eval = 0
            for _ in range(n_sim):
                p, i = draw(self.Model)
                n_eval += 1
                res.append(p)
            res.keep('N_Eval', n_eval)
        res.finish()

        self.info('Finish')


if __name__ == '__main__':
    from sims_pars.fitting.cases import BetaBin

    model0 = BetaBin()

    alg = PriorSampling(parallel=True)

    alg.fit(model0)
    res_prior = alg.Collector

    print(res_prior.DF.describe())
