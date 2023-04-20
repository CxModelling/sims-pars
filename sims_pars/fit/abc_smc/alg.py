from sims_pars.fit.base import Fitter, Particle
from sims_pars.fit.results import ParameterSet
import numpy as np
import numpy.random as rd
from joblib import Parallel, delayed
from tqdm import tqdm

__author__ = 'Chu-Chang Ku'
__all__ = ['ApproxBayesComSMC']


class StateABCSMC:
    def __init__(self, wts, pts, rnd=0, stay=0, ess=0, acc=1, eps=np.Inf, eps_thres=np.Inf):
        self.Wts = wts
        self.Particles = pts
        self.Round = rnd
        self.Stay = stay
        self.ESS = ess
        self.Acceptance = acc
        self.Eps = eps
        self.EpsThres = eps_thres


def sample_fin(model, unpack=False):
    di = np.Inf
    n_eval = 0
    while np.isinf(di):
        n_eval += 1
        p = model.sample_prior()
        sim = model.simulate(p)
        di = model.calc_distance(sim)

    sim = sim.to_json() if unpack else sim
    return di, sim, n_eval


def sample_ess(model, eps, unpack=False):
    di = np.Inf
    n_eval = 0
    while di > eps:
        n_eval += 1
        p = model.sample_prior()
        sim = model.simulate(p)
        di = model.calc_distance(sim)

    sim = sim.to_json() if unpack else sim
    return di, sim, n_eval


def mutate(p0, steps):
    sizes = {k: np.random.normal(0, v) for k, v in steps.items()}
    return {k: p0[k] + size for k, size in sizes.items()}


def sample_mutation(model, pt0, steps, unpack=False):
    di = np.Inf
    n_eval = 0
    p0 = pt0.Pars if isinstance(pt0, Particle) else pt0['Pars']['Locus']
    while np.isinf(di):
        p1 = mutate(p0, steps)
        p1 = model.serve(p1)
        if np.isinf(p1.LogProb):
            continue

        n_eval += 1

        sim = model.simulate(p1)
        di = model.calc_distance(sim)

        if n_eval > 20:
            sim = pt0
            di = pt0['distance']
            break

    sim = sim.to_json() if unpack else sim
    return di, sim, n_eval


def test(pts):
    po = ParameterSet('')
    for pt in pts:
        po.append(pt)
    print(po.to_df().describe())


class ApproxBayesComSMC(Fitter):
    def __init__(self, **kwargs):
        Fitter.__init__(self, 'ApproxBayesComSMC', **kwargs)

    @property
    def DefaultSettings(self) -> dict:
        return {
            'n_iter': 500,
            'parallel': True,
            'alpha': 0.9,
            'p_thres': 0.6,
            'max_round': 20,
            'max_stay': 3,
            'min_acc': 0.1,
            'n_core': 4,
            'verbose': 5
        }

    def initialise(self):
        self.info("Initialising")

        n_sim = self.Settings['n_iter']

        if self.Settings['parallel']:
            with Parallel(n_jobs=self.Settings['n_core'], verbose=self.Settings['verbose']) as parallel:
                samples = parallel(delayed(sample_fin)(self.Model, unpack=True) for _ in range(n_sim))
                samples = [(di, Particle.from_json(sim), ne) for di, sim, ne in samples]
        else:
            samples = [sample_fin(self.Model) for _ in tqdm(range(n_sim))]

        pts = [pt for _, pt, _ in samples]
        wts = np.ones(n_sim) / n_sim
        n_eval = sum(i for _, _, i in samples)
        ess = 1 / sum(wts * wts)

        eps = np.inf

        self.Monitor.keep(Round=0, Eval=n_eval, Eps=eps, ESS=ess, Acc=1)
        self.Monitor.step()

        self.State = StateABCSMC(
            wts, pts, acc=1,
            rnd=0, stay=0, ess=ess, eps=eps, eps_thres=n_sim * self.Settings['p_thres']
        )
        self.info(f'Round 0, ESS {ess:.2f}')

    def find_eps(self, eps0, theta0):
        ds = np.array([self.Model.calc_distance(pt) for pt in theta0])

        e0 = (ds <= eps0).mean()
        et = self.Settings['alpha'] * e0

        return np.quantile(ds, et)

    def update(self, **kwargs):
        while True:
            self._a_round()
            if self._check_termination():
                break

    def _a_round(self):
        state = self.State
        rnd = state.Round + 1
        eps0, theta0, wts = state.Eps, state.Particles, state.Wts

        eps1 = self.find_eps(eps0, theta0)
        if eps1 >= eps0:
            stay = state.Stay + 1
            eps1 = eps0
        else:
            stay = 0

        wts = self.update_wts(eps0, theta0, wts, eps1)
        wts, theta1 = self.resample(theta0, wts, state.EpsThres)
        pts, ess, acc, ne = self.mcmc_proposal(theta1, wts, eps1)
        #
        # test(theta1)
        # test(pts)

        self.State = StateABCSMC(
            wts, pts, acc=acc,
            rnd=rnd, stay=stay, ess=ess, eps=eps1, eps_thres=state.EpsThres
        )

        self.Monitor.keep(Round=rnd, Eval=ne, Eps=eps1, ESS=ess, Acc=acc)
        self.Monitor.step()
        self.info(f'Round {rnd}, ESS {ess:.0f}, Epsilon {eps1:.4f}, Acceptance {acc:.1%}')

    def update_wts(self, eps0, theta0, wts0, eps1):
        ds = [self.Model.calc_distance(p) for p in theta0]

        wts = wts0.copy()
        for a in range(len(wts)):
            d = ds[a]
            if d < eps0:
                wts[a] = wts[a] if d < eps1 else 0
        wts /= wts.sum()
        return wts

    def resample(self, theta0, wts, eps_thres):
        assert sum(wts > 0) > 2

        n_sim = self.Settings['n_iter']

        if eps_thres * (wts ** 2).sum() > 1:
            ind = np.where(wts > 0, wts, 0)
            ind /= ind.sum()

            theta1 = np.random.choice(list(range(len(theta0))), n_sim, replace=True, p=ind)
            theta1 = [theta0[i] for i in theta1]
            wts = np.ones(n_sim) / n_sim
        else:
            theta1 = theta0
        theta1 = [theta.copy() for theta in theta1]
        return wts, theta1

    def mcmc_proposal(self, theta1, wts, eps1):
        n_iter = self.Settings['n_iter']
        tau = self.calc_weighted_std(theta1, wts)
        if self.Settings['parallel']:
            with Parallel(n_jobs=self.Settings['n_core'], verbose=self.Settings['verbose']) as parallel:
                sample_p = parallel(delayed(sample_mutation)(self.Model, pt.to_json(), tau, unpack=True) for pt in theta1)
            sample_p = [(di, Particle.from_json(sim), i) for di, sim, i in sample_p]
        else:
            sample_p = [sample_mutation(self.Model, pt, tau) for pt in tqdm(theta1)]

        theta_p = [sim for _, sim, _ in sample_p]
        n_eval = sum([i for _, _, i in sample_p])

        # MH acceptance ratio
        acc = np.zeros(n_iter)
        for i, p in enumerate(theta_p):
            d_p = p['distance']
            if d_p < eps1:
                acc[i] = 1

        # Update accepted proposals
        pts = list()

        accepted = 0
        for i in range(n_iter):
            if rd.random() < acc[i]:
                p = theta_p[i]
                accepted += 1
            else:
                p = theta1[i]
            pts.append(p)

        ess = 1 / sum(wts * wts)
        acc = accepted / n_iter

        return pts, ess, acc, n_eval

    def calc_weighted_std(self, theta1, wts):
        tau = dict()

        for dom in self.Model.Domain:
            key = dom.Name
            vs = np.array([p.Pars[key] for p in theta1])
            average = np.average(vs, weights=wts)
            variance = np.average((vs - average) ** 2, weights=wts)

            if variance > 0:
                tau[key] = np.sqrt(variance)
        return tau

    def _check_termination(self):
        rnd, stay, acc = self.State.Round, self.State.Stay, self.State.Acceptance

        if rnd >= self.Settings['max_round']:
            return True
        elif rnd >= 3 and stay >= self.Settings['max_stay']:
            self.info('Early terminated due to convergence')
            return True
        elif acc < self.Settings['min_acc']:
            self.info('Early terminated due to low acceptance')
            return True
        return False

    def terminate(self):
        pass

    def sample_posteriors(self, n_collect=300) -> ParameterSet:
        assert self.State is not None
        self.info('Collecting posterior')
        state = self.State
        wts, eps, pts = state.Wts, state.Eps, state.Particles

        ind = np.where(wts > 0, wts, 0)
        ind /= ind.sum()

        theta1 = np.random.choice(list(range(len(pts))), n_collect, replace=True, p=ind)
        theta1 = [pts[i] for i in theta1]

        tau = self.calc_weighted_std(pts, wts)

        if self.Settings['parallel']:
            with Parallel(n_jobs=self.Settings['n_core'], verbose=self.Settings['verbose']) as parallel:
                sample_p = parallel(delayed(sample_mutation)(self.Model, pt.to_json(), tau, unpack=True) for pt in theta1)
            sample_p = [(di, Particle.from_json(sim), i) for di, sim, i in sample_p]

        else:
            sample_p = [sample_mutation(self.Model, pt, tau) for pt in tqdm(theta1)]

        theta_p = [sim for _, sim, _ in sample_p]

        post = ParameterSet('ABC-SMC')

        for pt1, ptp in zip(theta1, theta_p):
            d_p = ptp['distance']
            if d_p < eps:
                post.append(pt1)
            else:
                post.append(pt1)

        post.keep('Trace', self.Monitor.Trajectories)
        post.keep('Eps', eps)

        return post


if __name__ == '__main__':
    from sims_pars.fit.toys import get_sir

    m0 = get_sir(be=1.5, ga=0.2)

    alg = ApproxBayesComSMC(n_iter=300, max_round=30, parallel=True, verbose=False)
    alg.fit(m0)

    po = alg.sample_posteriors(1000)

    print(po.to_df().describe())
