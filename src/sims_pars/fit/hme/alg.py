from sims_pars.fit.converter import *
from sims_pars.fit.base import *
from sims_pars.fit.results import ParameterSet
import numpy as np
import numpy.random as rd
from scipy.stats.qmc import LatinHypercube
import tqdm

__author__ = 'Chu-Chang Ku'
__all__ = ['BayesHistoryMatching']


class StateHME:
    def __init__(self, ems, cns, mts, pts, bounds, wave=0, mse=np.inf, cutoff=np.inf, acc=1.0):
        self.Emulators = ems
        self.Converters = cns
        self.Matchers = mts
        self.Particles = pts
        self.Bounds = bounds
        self.CutOff = cutoff
        self.MSE = mse
        self.Wave = wave
        self.Acc = acc


class Matchers:
    def __init__(self, v_m, data):
        self.VarModel = v_m
        self.Data = dict(data)

    def eval_implausibility(self, pt):
        ims = list()

        for k, d in self.Data.items():
            e, v_o = d.ev()
            y, v_e = pt['Ys'][k], pt['Vs'][k]
            imp = np.abs(e - y) / np.sqrt(self.VarModel + v_o + v_e)
            ims.append(float(imp))

        return np.max(ims)

    def eval_mse(self, pts: list[Particle]):
        residuals = [np.mean([np.power(pt['Ys'][k] - v.Value, 2) for k, v in self.Data.items()]) for pt in pts]
        return np.mean(residuals)


class BayesHistoryMatching(Fitter):
    def __init__(self, **kwargs):
        Fitter.__init__(self, 'BHM', **kwargs)

    @property
    def DefaultSettings(self) -> dict:
        return {
            'n_sims': 0,
            'n_ems': 0,
            'n_pars': 0,
            'max_wave': 5,
            'var_model': 1,
            'cutoff': 3,
            'min_acc': 0.2
        }

    def initialise(self):
        from sims_pars.fit.hme.emulator import GPREmulator

        self.info('Initialising')
        model = self.Model
        self.Settings['n_pars'] = n_pars = len(model.Domain)

        if self.Settings['n_sims'] <= 0:
            self.Settings['n_sims'] = n_sims = n_pars * 10

        if self.Settings['n_ems'] <= 0:
            self.Settings['n_ems'] = n_sims * 100

        v_m = self.Settings['var_model']

        bnd = np.zeros(n_pars), np.ones(n_pars)
        converters = [get_converter(dom) for dom in model.Domain]
        emulators = {k: GPREmulator(k) for k, v in model.Data.items()}
        matchers = Matchers(v_m, model.Data)

        self.State = StateHME(emulators, converters, matchers, [], bnd)

    def update(self):
        while True:
            self._a_wave()
            if self._check_termination():
                break

    def _a_wave(self):
        state = self.State
        self.info(f'Wave: {state.Wave + 1}')

        pts_sim = self._simulate(state)
        pts_emu = self._emulate(pts_sim, state)
        pts_pls, co, acc = self._keep_plausible(pts_emu)

        mse = state.Matchers.eval_mse(pts_pls)
        self.info(f'CutOff: {co}, Acceptance: {acc}, MSE: {mse}')

        xss = np.array([pt['Xs'] for pt in pts_pls])
        bnd = xss.min(0), xss.max(0)

        self.State = StateHME(state.Emulators, state.Converters, state.Matchers, pts_pls, bnd,
                              wave=state.Wave + 1, mse=mse, cutoff=co, acc=acc)

    def _simulate(self, state):
        lower, upper = state.Bounds
        cons = state.Converters
        model = self.Model
        n_pars, n_sims = self.Settings['n_pars'], self.Settings['n_sims']
        xss = LatinHypercube(n_pars).random(n_sims)
        xss = lower + xss * (upper - lower)
        pts_sim = list()

        for xs in tqdm.tqdm(xss, desc='Simulation'):
            pars = {dom.Name: con.uniform2value(x) for dom, con, x in zip(model.Domain, cons, xs)}
            sim = model.simulate(pars)
            model.flatten(sim)
            pts_sim.append(sim)
        return pts_sim

    def _emulate(self, pts_sim, state):
        emulators, matchers, cons = state.Emulators, state.Matchers, state.Converters
        n_pars, n_ems = self.Settings['n_pars'], self.Settings['n_ems']
        model = self.Model

        xss = [pt.Notes['Xs'] for pt in pts_sim]
        yss = [pt.Notes['Ys'] for pt in pts_sim]

        self.info('Train emulators')
        for emu in emulators.values():
            emu.train(xss, yss)
        lower, upper = state.Bounds
        xss = lower + rd.random((n_ems, n_pars)) * (upper - lower)
        yss = {k: emu.predict(xss) for k, emu in emulators.items()}

        pts_emu = list()

        for i in tqdm.tqdm(range(n_ems), desc='Emulation'):
            xs = xss[i]
            pars = {dom.Name: con.uniform2value(x) for dom, con, x in zip(model.Domain, cons, xs)}
            sims = {k: ys[0][i] for k, ys in yss.items()}
            vs = {k: ys[1][i] for k, ys in yss.items()}

            pt = Particle(pars, sims)
            pt['Vs'] = vs
            pt['Xs'] = xs
            pt['Ys'] = sims
            pt['Imp'] = matchers.eval_implausibility(pt)
            pts_emu.append(pt)
        return pts_emu

    def _keep_plausible(self, pts_emu):
        ims = [pt['Imp'] for pt in pts_emu]

        co, acc = self.Settings['cutoff'], self.Settings['min_acc']
        co = max(co, np.quantile(ims, acc))
        pts_pls = [pt for pt in pts_emu if pt['Imp'] < co]

        acc = len(pts_pls) / len(pts_emu)
        return pts_pls, co, acc

    def _check_termination(self):
        if self.Settings['max_wave'] <= self.State.Wave:
            self.info('Termination, max. wave reached')
            return True
        # if self.Settings['cutoff'] < self.State.CutOff:
        #     return False
        return False

    def terminate(self):
        pass

    def sample_posteriors(self, n_collect=300):
        assert self.State is not None

        state = self.State
        n_pars = self.Settings['n_pars']
        lower, upper = state.Bounds
        cons = state.Converters
        model = self.Model
        xss = lower + rd.random((n_collect, n_pars)) * (upper - lower)

        post = ParameterSet('HME')

        for xs in xss:
            pars = {dom.Name: con.uniform2value(x) for dom, con, x in zip(model.Domain, cons, xs)}
            sim = model.simulate(pars)
            model.flatten(sim)
            post.append(sim)

        bounds = [(dom.Name, con.uniform2value(lo), con.uniform2value(up)) for dom, con, lo, up in zip(model.Domain, cons, lower, upper)]

        post.keep('MSE', state.MSE)
        post.keep('Acceptance', state.Acc)
        post.keep('CutOff', state.CutOff)
        post.keep('Bounds', bounds)

        return post


if __name__ == '__main__':
    from sims_pars.fit.toys import get_betabin

    m0 = get_betabin([4, 12])

    alg = BayesHistoryMatching(max_wave=3)
    alg.fit(m0)

    po = alg.sample_posteriors()

    for bd in po['Bounds']:
        print(bd)

    print(po.to_df().describe())
