from sims_pars.fitting.fitter import Fitter, ParameterSet
from sims_pars.fitting.util import draw, draw_parallel, serve_and_evaluate, serve_and_evaluate_parallel
from sims_pars.fitting.ga.cross import ShuffleCrossover
import numpy as np
from joblib import Parallel
from tqdm import tqdm
from pydantic import BaseModel
from typing import Any


__author__ = 'Chu-Chang Ku'
__all__ = ['GeneticAlg']


class States(BaseModel):
    Generation = 0
    Stay = 0
    MaxFitness = - np.inf
    MeanFitness = - np.inf
    Best: Any = None


class GeneticAlg(Fitter):
    def __init__(self, **kwargs):
        Fitter.__init__(self, 'GeneticAlg', **kwargs)

        self.States = States()

        self.Mutators = list()
        self.Crossover = ShuffleCrossover()


    @property
    def DefaultSettings(self) -> dict:
        return {
            'n_collect': 300,
            'parallel': True,
            'max_round': 100,
            'max_stay': 3,
            'n_core': 4,
            'verbose': 5,
            'p_mut': 0.1,
            'p_cro': 0.8,
            'mut': 'rw()',
            'cro': 'shuffle()',
            'sel': 'importance()'
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
                samples = draw_parallel(self.Model, n_sim, parallel)
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

    def update(self, **kwargs):
        self.info('Start updating')
        self.States.Stay = 0


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


    def collect(self, **kwargs):
        self.info("Collecting posteriors")
        self.Collector.keep('Trace', self.Monitor.Trajectories)
        self.Collector.finish()


    def __genesis(self, n):
        for _ in range(n):
            p = self.Model.sample_prior()
            p.LogLikelihood = self.Model.evaluate_likelihood(p)
            self.Population.append(p)

    def __crossover(self):
        pop = self.Population
        n = len(pop)
        sel = rd.binomial(1, self.p_crossover, int(n / 2))

        for i, s in enumerate(sel):
            if s:
                p1, p2 = self.Crossover.crossover(pop[i * 2], pop[i * 2 + 1], self.Model.BN)
                self.Model.evaluate_prior(p1)
                self.Model.evaluate_prior(p2)
                p1.LogLikelihood = self.Model.evaluate_likelihood(p1)
                p2.LogLikelihood = self.Model.evaluate_likelihood(p2)
                pop[i * 2], pop[i * 2 + 1] = p1, p2

    def __mutation(self):
        for node, mut in zip(self.Moveable, self.Mutators):
            i = node['Name']
            vs = [gene[i] for gene in self.Population]
            mut.set_scale(vs)

        pop = self.Population
        n = len(pop)
        sel = rd.binomial(1, self.p_mutation, n)

        for i, s in enumerate(sel):
            if s:
                p = pop[i] = pop[i].clone()
                loc = dict()
                for mut in self.Mutators:
                    loc[mut.Name] = mut.proposal(p[mut.Name])
                p.impulse(loc, self.Model.BN)
                self.Model.evaluate_prior(p)
                p.LogLikelihood = self.Model.evaluate_likelihood(p)

    def __selection(self):
        for p in self.Population:
            if p.LogLikelihood is 0:
                p.LogLikelihood = self.Model.evaluate_likelihood(p)

        if self.Target == 'MAP':
            wts = [p.LogPosterior for p in self.Population]
        else:
            wts = [p.LogLikelihood for p in self.Population]
        pop, mean = resample(wts, self.Population)
        self.Population = [p.clone() for p in pop]
        self.MeanFitness = mean

    def __find_elitism(self):
        if self.Target == 'MAP':
            self.BestFit = max(self.Population, key=lambda x: x.LogPosterior)
        else:
            self.BestFit = max(self.Population, key=lambda x: x.LogLikelihood)

        fitness = self.BestFit.LogPosterior if self.Target == 'MAP' else self.BestFit.LogLikelihood

        if fitness == self.MaxFitness:
            self.Stay += 1

        self.MaxFitness = fitness
        self.Series.append({
            'Generation': self.Generation,
            'Max fitness': self.MaxFitness,
            'Mean fitness': self.MeanFitness
        })
        self.info('Generation: {}, Mean fitness: {:.2E}, Max fitness: {:.2E}'.format(
            self.Generation, self.MeanFitness, self.MaxFitness))

    def __termination(self):
        if self.Stay > 5:
            return True


if __name__ == '__main__':
    from sims_pars.fitting.cases import BetaBin
    from sims_pars.fitting.fitter import PriorSampling

    model0 = BetaBin()
    print('Free parameters: ', model0.FreeParameters)

    alg = PriorSampling(parallel=True, n_collect=200)

    alg.fit(model0)
    res_post = alg.Collector

    print(res_post.DF[['p1', 'p2']].describe())

    alg = ApproxBayesComSMC(parallel=True, n_collect=300, max_round=10)

    alg.fit(model0)
    res_post = alg.Collector

    print(res_post.DF[['p1', 'p2']].describe())
    print(alg.Monitor.Trajectories)
