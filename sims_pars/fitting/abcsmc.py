from sims_pars.fitting.fitter import Fitter, ParameterSet
from sims_pars.fitting.util import *
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

__author__ = 'Chu-Chang Ku'
__all__ = ['ApproxBayesComSMC']


class ApproxBayesComSMC(Fitter):
    def __init__(self, **kwargs):
        Fitter.__init__(self, 'ApproxBayesCom', **kwargs)

        self.EpsThres = self.Settings['n_collect'] * self.Settings['p_thres']
        self.Eps = np.inf
        self.N_Round = 0
        self.N_Stay = 0
        self.wts = np.ones(self.Settings['n_collect'])

    @property
    def DefaultSettings(self) -> dict:
        return {
            'n_collect': 1000,
            'parallel': True,
            'alpha': 0.9,
            'p_thres': 0.6,
            'max_round': 20,
            'max_stay': 3,
            'n_core': 4,
            'verbose': 0
        }

    def initialise(self):
        self.Collector = post = ParameterSet('Test')

        self.info("Initialising")

        self.Eps = np.inf
        self.N_Round = 0
        self.N_Stay = 0

        n_sim = self.Settings['n_collect']

        if self.Settings['parallel']:
            with Parallel(n_jobs=self.Settings['n_core'], verbose=self.Settings['verbose']) as parallel:
                samples = parallel(delayed(draw)(self.Model, unpack=True) for _ in range(n_sim))
            samples = [(self.Model.serve_from_json(p), i) for p, i in samples]
        else:
            samples = [draw(self.Model) for _ in tqdm(range(n_sim))]

        for p, _ in samples:
            post.append(p)

        self.wts = np.ones(n_sim) / n_sim
        n_eval = sum(i for _, i in samples)
        ess = 1 / sum(self.wts * self.wts)

        self.Monitor.keep(Round=self.N_Round, Eval=n_eval, Eps=self.Eps, ESS=ess, Acc=1)
        self.Monitor.step()
        self.info(f'Round 0, ESS {ess:.2f}')

    def find_eps(self):
        eps0 = self.Eps

        ds = [- p.LogLikelihood for p in self.Collector.ParameterList]
        ds = np.array(ds)

        e0 = (ds < eps0).mean()
        et = self.Settings['alpha'] * e0

        return np.quantile(ds, et)

    def update(self, **kwargs):
        while True:
            self.N_Round += 1
            eps0, eps1 = self.Eps, self.find_eps()
            if eps1 > eps0:
                self.N_Stay += 1
                eps1 = eps0
            else:
                self.N_Stay = 0

            self.Eps = eps1
            self.update_wts(eps0, eps1)
            theta1 = self.resample()
            self.mcmc_proposal(theta1, eps1)

            if self.N_Round >= self.Settings['max_round']:
                break
            elif self.N_Round >= 3 and self.N_Stay >= self.Settings['max_stay']:
                self.info('Early terminated due to convergence')
                break

    def update_wts(self, eps0, eps1):
        ds = [- p.LogLikelihood for p in self.Collector.ParameterList]
        for a in range(len(self.wts)):
            d = ds[a]
            if d < eps0:
                self.wts[a] = self.wts[a] if d < eps1 else 0
        self.wts /= self.wts.sum()

    def resample(self):
        assert sum(self.wts > 0) > 2

        theta0 = list(self.Collector.ParameterList)
        n_sim = self.Settings['n_collect']

        if self.EpsThres * (self.wts ** 2).sum() > 1:
            ind = np.where(self.wts > 0, self.wts, 0)
            ind /= ind.sum()

            theta1 = np.random.choice(list(range(len(theta0))), n_sim, replace=True, p=ind)
            theta1 = [theta0[i] for i in theta1]
            self.wts = np.ones(n_sim) / n_sim
        else:
            theta1 = theta0
        theta1 = [theta.clone() for theta in theta1]
        return theta1

    def mcmc_proposal(self, theta1, eps1):
        n_collect = self.Settings['n_collect']
        tau = self.calc_weighted_std(theta1)

        if self.Settings['parallel']:
            parallel = Parallel(n_jobs=self.Settings['n_core'], verbose=self.Settings['verbose'])
            sample_p = mutate_and_draw_parallel(self.Model, theta1, tau, parallel)
        else:
            sample_p = [mutate_and_draw(self.Model, p, tau) for p in tqdm(theta1)]

        theta_p = [p for p, _ in sample_p]
        n_eval = sum([i for _, i in sample_p])

        # MH acceptance ratio
        acc = np.zeros(n_collect)
        for i, p in enumerate(theta_p):
            d_p = - p.LogLikelihood
            if d_p < eps1:
                acc[i] = 1

        # Update accepted proposals
        self.Collector = post = ParameterSet()

        accepted = 0
        for i in range(n_collect):
            if np.random.random() < acc[i]:
                p = theta_p[i]
                accepted += 1
            else:
                p = theta1[i]
            post.append(p)

        ess = 1 / sum(self.wts * self.wts)
        acc = accepted / n_collect

        self.Monitor.keep(Round=self.N_Round, Eval=n_eval, Eps=self.Eps, ESS=ess, Acc=acc)
        self.Monitor.step()
        self.info(f'Round {self.N_Round}, ESS {ess:.0f}, Epsilon {self.Eps:.4f}, Acceptance {acc:.1%}')

    def calc_weighted_std(self, theta1):
        tau = dict()

        for key in self.Model.FreeParameters:
            vs = np.array([p[key] for p in theta1])
            average = np.average(vs, weights=self.wts)
            variance = np.average((vs - average) ** 2, weights=self.wts)

            if variance > 0:
                tau[key] = np.sqrt(variance)
        return tau

    def collect(self, **kwargs):
        self.info("Collecting posteriors")
        self.Collector.keep('Trace', self.Monitor.Trajectories)
        self.Collector.finish()


if __name__ == '__main__':
    from sims_pars.fitting.cases import BetaBin
    from sims_pars.fitting.fitter import PriorSampling

    model0 = BetaBin()

    alg = PriorSampling(parallel=True, n_collect=300)

    alg.fit(model0)
    res_post = alg.Collector

    print(res_post.DF[['p1', 'p2']].describe())

    alg = ApproxBayesComSMC(parallel=True, n_collect=300, max_round=10)

    alg.fit(model0)
    res_post = alg.Collector

    print(res_post.DF[['p1', 'p2']].describe())
    print(alg.Monitor.Trajectories)
