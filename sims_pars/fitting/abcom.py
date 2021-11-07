from sims_pars.fitting.fitter import Fitter, ParameterSet
from sims_pars.fitting.util import draw
import numpy as np
from joblib import Parallel, delayed

__author__ = 'Chu-Chang Ku'
__all__ = ['ApproxBayesCom']


def pick(obj, eps, unpack=True):
    p, i = draw(obj, unpack=False)
    while p.LogLikelihood < eps:
        p, i0 = draw(obj, unpack=False)
        i += i0

    if unpack:
        p = p.to_json()
    return p, i


class ApproxBayesCom(Fitter):
    def __init__(self, **kwargs):
        Fitter.__init__(self, 'ApproxBayesCom', **kwargs)

    @property
    def DefaultSettings(self) -> dict:
        return {
            'n_collect': 1000,
            'parallel': True,
            'n_test': 200,
            'p_test': 0.95,
            'n_core': 4,
            'verbose': 5
        }

    def initialise(self):
        self.Collector = prior = ParameterSet('Test')

        n_sim = self.Settings['n_test']

        if self.Settings['parallel']:
            self.info('Start a parallel sampler for collecting test runs')
            with Parallel(n_jobs=self.Settings['n_core'], verbose=self.Settings['verbose']) as parallel:
                samples = parallel(delayed(draw)(self.Model, unpack=True) for _ in range(n_sim))

            for p, _ in samples:
                p = self.Model.serve_from_json(p)
                prior.append(p)
            n_eval = sum(i for _, i in samples)

        else:
            self.info('Start a sampler for collecting test runs')

            n_eval = 0
            for _ in range(n_sim):
                p, i = draw(self.Model)
                n_eval += 1
                prior.append(p)
        li = [p.LogLikelihood for p in prior.ParameterList]

        eps = np.quantile(np.array(li), self.Settings['p_test'])
        prior.keep('Prior_Draw', n_eval)
        prior.keep('Prior_Collect', n_sim)
        prior.keep('Prior_Yield', n_sim / n_eval)
        prior.keep('Eps', - eps)

        self.info(f'Eps: {-eps:g}')

        prior.finish()

    def update(self):
        pass

    def collect(self):
        prior = self.Collector
        self.Collector = post = ParameterSet('Posterior')
        post.Notes.update(prior.Notes)

        eps = - prior['Eps']
        n_sim = self.Settings['n_collect']

        if self.Settings['parallel']:
            self.info('Start a parallel sampler for collecting posterior runs')
            with Parallel(n_jobs=self.Settings['n_core'], verbose=self.Settings['verbose']) as parallel:
                samples = parallel(delayed(pick)(self.Model, eps=eps, unpack=True) for _ in range(n_sim))

            for p, _ in samples:
                p = self.Model.serve_from_json(p)
                post.append(p)
            n_eval = sum(i for _, i in samples)
        else:
            self.info('Start a sampler for collecting posterior runs')

            n_eval = 0
            for _ in range(n_sim):
                p, i = pick(self.Model, eps)
                n_eval += 1
                post.append(p)

        post.keep('Posterior_Draw', n_eval)
        post.keep('Posterior_Collect', n_sim)
        post.keep('Posterior_Yield', n_sim / n_eval)

        post.finish()


if __name__ == '__main__':
    from sims_pars.fitting.cases import BetaBin
    from sims_pars.fitting.fitter import PriorSampling
    import logging

    model0 = BetaBin()

    alg = PriorSampling(parallel=True)
    alg.Monitor.add_handler(logging.StreamHandler())

    alg.fit(model0)
    res_post = alg.Collector

    print(res_post.DF[['p1', 'p2']].describe())

    alg = ApproxBayesCom(parallel=True)
    alg.Monitor.add_handler(logging.StreamHandler())

    alg.fit(model0)
    res_post = alg.Collector

    print(res_post.DF[['p1', 'p2']].describe())
    print(res_post.Notes)
