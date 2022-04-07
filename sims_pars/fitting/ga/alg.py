from sims_pars.fitting.fitter import Fitter, ParameterSet
from sims_pars.fitting.util import draw, draw_parallel, serve_and_evaluate, serve_and_evaluate_parallel
from sims_pars.fitting.ga.cross import get_crossover
from sims_pars.fitting.ga.select import get_selector
from sims_pars.fitting.ga.mutate import get_mutator
import numpy as np
import numpy.random as rd
from scipy.special import logsumexp
from joblib import Parallel
from tqdm import tqdm
from pydantic import BaseModel
from typing import Any


__author__ = 'Chu-Chang Ku'
__all__ = ['GeneticAlg']


class States(BaseModel):
    Generation: int = 0
    Stay: int = 0
    MaxFitness: float = - np.inf
    MeanFitness: float = - np.inf
    Best: Any = None


class GeneticAlg(Fitter):
    def __init__(self, **kwargs):
        Fitter.__init__(self, 'GeneticAlg', **kwargs)

        self.States = States()
        self.Crossover = get_crossover(self.Settings['cro'])
        self.Mutator = get_mutator(self.Settings['mut'])
        self.Selector = get_selector(self.Settings['sel'])

    @property
    def DefaultSettings(self) -> dict:
        return {
            'n_collect': 300,
            'parallel': True,
            'max_round': 100,
            'max_stay': 5,
            'n_core': 4,
            'verbose': 5,
            'p_mut': 0.1,
            'p_cro': 0.8,
            'mut': 'rw',
            'cro': 'shuffle',
            'sel': 'importance',
            'target': 'MAP'
        }

    def initialise(self):
        self.info("Initialising")
        self.Collector = ParameterSet('Test')
        self.States = States()

        self.__genesis()
        self.__find_elitism()

    def update(self, **kwargs):
        self.info('Start updating')
        while True:
            self.States.Generation += 1
            self.__crossover()
            self.__mutation()
            self.__selection()
            self.__find_elitism()
            if self.__termination():
                break

    def collect(self, **kwargs):
        self.info("Collecting posteriors")
        self.Collector.keep('Trace', self.Monitor.Trajectories)
        self.Collector.finish()

    def __genesis(self):
        self.info('Genesis')
        post = self.Collector.ParameterList
        n_sim = self.Settings['n_collect']

        if self.Settings['parallel']:
            with Parallel(n_jobs=self.Settings['n_core'], verbose=self.Settings['verbose']) as parallel:
                samples = draw_parallel(self.Model, n_sim, parallel)
        else:
            samples = [draw(self.Model) for _ in tqdm(range(n_sim))]

        for p, _ in samples:
            post.append(p)

    def __crossover(self):
        ps = self.Collector.ParameterList
        n = len(ps)
        sel = rd.binomial(1, self.Settings['p_cro'], int(n / 2))
        sel, = np.where(sel)

        for i in sel:
            p1, p2 = self.Crossover.crossover(ps[i * 2], ps[i * 2 + 1], self.Model)
            ps[i * 2] = serve_and_evaluate(self.Model, p1)
            ps[i * 2 + 1] = serve_and_evaluate(self.Model, p2)

    def __mutation(self):
        ps = self.Collector.ParameterList
        self.Mutator.set_scales(ps, self.Model)
        n = len(ps)

        nxt,  = np.where(rd.binomial(1, self.Settings['p_mut'], n))
        proposed = [self.Mutator.propose(p, self.Model) for p in ps]

        if self.Settings['parallel']:
            with Parallel(n_jobs=self.Settings['n_core'], verbose=self.Settings['verbose']) as parallel:
                proposed = serve_and_evaluate_parallel(self.Model, proposed, parallel)
        else:
            proposed = [serve_and_evaluate(self.Model, p) for p in tqdm(proposed)]

        for i, p in zip(nxt, proposed):
            if np.isfinite(p.LogLikelihood):
                ps[i] = p

    def __selection(self):
        ps1 = self.Selector.select(self.Collector.ParameterList, self.Model)

        self.Collector = ParameterSet()
        for p in ps1:
            self.Collector.append(p)

    def __find_elitism(self):
        states = self.States
        fitness = states.MaxFitness
        if self.Settings['target'] == 'MAP':
            states.Best = max(self.Collector.ParameterList, key=lambda x: x.LogPosterior)
            states.MaxFitness = states.Best.LogPosterior
            wts = [p.LogPosterior for p in self.Collector.ParameterList]
        else:
            states.Best = max(self.Collector.ParameterList, key=lambda x: x.LogLikelihood)
            states.MaxFitness = states.Best.LogLikelihood
            wts = [p.LogLikelihood for p in self.Collector.ParameterList]

        states.MeanFitness = logsumexp(wts) - np.log(len(wts))

        if fitness >= states.MaxFitness:
            states.Stay += 1
        else:
            states.Stay = 0

        self.Monitor.keep(Round=states.Generation, Stay=states.Stay, MaxFitness=states.MaxFitness, MeanFitness=states.MeanFitness)
        self.Monitor.step()
        self.info(f'Round {states.Generation}, Max fitness: {states.MaxFitness:.4g}, '
                  f'Mean fitness: {states.MeanFitness:.4g}')

    def __termination(self):
        if self.States.Stay > self.Settings['max_stay'] or self.States.Generation > self.Settings['max_round']:
            return True


if __name__ == '__main__':
    from sims_pars.fitting.cases import BetaBin

    model0 = BetaBin()
    print('Free parameters: ', model0.FreeParameters)

    alg = GeneticAlg(parallel=True, n_collect=200, max_round=50, verbose=0, sel='tour')

    alg.fit(model0)
    res_post = alg.Collector
    print(alg.States.Best)
    print(res_post.DF[['p1', 'p2']].describe())
    print(alg.Monitor.Trajectories)
