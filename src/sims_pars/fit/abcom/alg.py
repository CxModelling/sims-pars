from sims_pars.fit.base import *
from sims_pars.fit.results import ParameterSet
import numpy as np
import tqdm
from joblib import Parallel, delayed

__author__ = 'Chu-Chang Ku'
__all__ = ['ApproxBayesCom']


class StateABC:
    def __init__(self, eps, acc):
        self.Eps = eps
        self.Acc = acc


def sample_fin(model):
    di = np.Inf
    n_eval = 0
    while np.isinf(di):
        n_eval += 1
        p = model.sample_prior()
        sim = model.simulate(p)
        di = model.calc_distance(sim)
    return di, n_eval


def sample_ess(model, eps):
    di = np.Inf
    n_eval = 0
    while di > eps:
        n_eval += 1
        p = model.sample_prior()
        sim = model.simulate(p)
        di = model.calc_distance(sim)
    return di, sim.to_json(), n_eval


class ApproxBayesCom(Fitter):
    def __init__(self, **kwargs):
        Fitter.__init__(self, 'ApproxBayesCom', **kwargs)

    @property
    def DefaultSettings(self) -> dict:
        return {
            'parallel': True,
            'n_test': 200,
            'p_test': 0.05,
            'n_core': 4,
            'verbose': 5
        }

    def initialise(self):
        n_sim = self.Settings['n_test']

        self.info('Sample prior')

        if self.Settings['parallel']:
            self.info('Start a parallel sampler for collecting test runs')
            with Parallel(n_jobs=self.Settings['n_core'], verbose=self.Settings['verbose']) as parallel:
                samples = parallel(delayed(sample_fin)(self.Model) for _ in range(n_sim))
            n_eval = sum([ne for _, ne in samples])
            dis = [di for di, _ in samples]

        else:
            self.info('Start a sampler for collecting test runs')
            dis = list()
            n_eval = 0
            for _ in tqdm.tqdm(range(n_sim), 'Evaluate prior'):
                di = np.Inf
                while np.isinf(di):
                    n_eval += 1
                    p = self.Model.sample_prior()
                    sim = self.Model.simulate(p)
                    di = self.Model.calc_distance(sim)
                dis.append(di)

        eps = np.quantile(np.array(dis), self.Settings['p_test'])
        self.State = StateABC(eps, self.Settings['p_test'])

        self.info(f'Prior_Yield: {n_sim / n_eval: .2%}', )
        self.info(f'Eps: {eps:g}')

    def update(self):
        pass

    def terminate(self):
        pass

    def sample_posteriors(self, n_collect=300):
        assert self.State is not None

        state = self.State
        eps = state.Eps

        post = ParameterSet('HME')
        post.keep('Eps', eps)

        if self.Settings['parallel']:
            with Parallel(n_jobs=self.Settings['n_core'], verbose=self.Settings['verbose']) as parallel:
                samples = parallel(delayed(sample_ess)(self.Model, eps=eps) for _ in range(n_collect))
            n_eval = sum([ne for _, _, ne in samples])
            for _, sim, _ in samples:
                post.append(Particle.from_json(sim))
        else:
            n_eval = 0
            for _ in tqdm.tqdm(range(n_collect), 'Collect posterior'):
                _, sim, ne = sample_ess(self.Model, eps)
                post.append(sim)
                n_eval += ne

        for pt in post.Particles:
            self.Model.flatten(pt)

        post.keep('Eps', state.Eps)
        post.keep('Acceptance', n_collect / n_eval)
        post.keep('ESS', n_collect)

        return post


if __name__ == '__main__':
    from sims_pars.fit.toys import get_betabin

    m0 = get_betabin([4, 4])

    alg = ApproxBayesCom(parallel=True)
    alg.fit(m0)

    po = alg.sample_posteriors(300)
    print(po.Notes)

    print(po.to_df().describe())

    print(po.to_pred_df())
