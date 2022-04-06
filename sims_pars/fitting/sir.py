from sims_pars.fitting.fitter import Fitter, ParameterSet
from sims_pars.fitting.util import draw, draw_parallel
import numpy as np
import numpy.random as rd
from scipy.special import logsumexp
from joblib import Parallel
from collections import Counter

__author__ = 'Chu-Chang Ku'
__all__ = ['SampImpResample']


def pick(obj, eps, unpack=True):
    p, i = draw(obj, unpack=False)
    while p.LogLikelihood < eps:
        p, i0 = draw(obj, unpack=False)
        i += i0

    if unpack:
        p = p.to_json()
    return p, i


class SampImpResample(Fitter):
    def __init__(self, **kwargs):
        Fitter.__init__(self, 'SampImpResample', **kwargs)

    @property
    def DefaultSettings(self) -> dict:
        return {
            'n_start': 10000,
            'n_collect': 10000,
            'parallel': True,
            'n_core': 4,
            'verbose': 5
        }

    def initialise(self):
        self.Collector = prior = ParameterSet('Test')

        n_sim = self.Settings['n_start']

        if self.Settings['parallel']:
            self.info('Start prior sampling with parallel computation')
            with Parallel(n_jobs=self.Settings['n_core'], verbose=self.Settings['verbose']) as parallel:
                samples = draw_parallel(self.Model, n_sim, parallel)
        else:
            self.info('Start prior sampling')
            samples = [draw(self.Model) for _ in range(n_sim)]

        for p, _ in samples:
            prior.append(p)
        n_eval = sum(i for _, i in samples)

        prior.keep('Prior_Draw', n_eval)
        prior.keep('Prior_Collect', n_sim)
        prior.keep('Prior_Yield', n_sim / n_eval)

        self.info(f'Yield: {n_sim} from {n_eval} evaluations {n_sim / n_eval:.2%}')

        prior.finish()

    def update(self):
        pass

    def collect(self):
        prior = self.Collector
        self.Collector = post = ParameterSet('Posterior')
        post.Notes.update(prior.Notes)

        ps = prior.ParameterList
        n_sim = self.Settings['n_collect']

        wt = np.array([p.LogLikelihood for p in ps], dtype=float)
        wt -= logsumexp(wt)

        resampled = rd.choice(len(prior), n_sim, replace=True, p=np.exp(wt))

        cnt = Counter(resampled)

        for i in resampled:
            post.append(ps[i].clone())

        wt = np.array([p.LogLikelihood for p in post.ParameterList], dtype=float)
        wt -= logsumexp(wt)
        wt = np.exp(wt)
        ess = np.power(wt.sum(), 2) / np.power(wt, 2).sum()

        post.keep('N_unique', len(cnt))
        post.keep('Max_repeated', max(cnt.values()))
        post.keep('ESS', ess)
        post.finish()


if __name__ == '__main__':
    from sims_pars.fitting.cases import BetaBin
    from sims_pars.fitting.fitter import PriorSampling

    model0 = BetaBin()

    # alg = PriorSampling(parallel=True, n_collect=200)
    # alg.fit(model0)
    # res_post = alg.Collector
    #
    # print(res_post.DF[['p1', 'p2']].describe())

    alg = SampImpResample(parallel=True, n_start=10000, n_collect=500)
    alg.fit(model0)
    res_post = alg.Collector

    print(res_post.DF[['p1', 'p2']].describe())
    print(res_post.Notes)
