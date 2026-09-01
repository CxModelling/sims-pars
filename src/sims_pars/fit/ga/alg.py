from sims_pars.fit.base import Fitter, Particle
from sims_pars.fit.results import ParameterSet
from sims_pars.fit.ga.util import sample_prior, evaluate_gene, fitness_of
from sims_pars.fit.ga.cross import get_crossover
from sims_pars.fit.ga.select import get_selector
from sims_pars.fit.ga.mutate import get_mutator
import numpy as np
import numpy.random as rd
from scipy.special import logsumexp
from joblib import Parallel, delayed
from tqdm import tqdm

__author__ = 'Chu-Chang Ku'
__all__ = ['GeneticAlg']


class StateGA:
    def __init__(self, pop, fitness, gen=0, stay=0, best=None,
                 max_fitness=-np.inf, mean_fitness=-np.inf):
        self.Population = pop          # list[Particle]
        self.Fitness = fitness         # np.ndarray, aligned with Population
        self.Generation = gen
        self.Stay = stay
        self.Best = best               # Particle
        self.MaxFitness = max_fitness
        self.MeanFitness = mean_fitness


class GeneticAlg(Fitter):
    """
    A real-coded genetic algorithm for MAP / maximum-goodness-of-fit point
    estimation, built on the same DataModel interface as ABC-SMC.
    """

    def __init__(self, **kwargs):
        Fitter.__init__(self, 'GeneticAlg', **kwargs)
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
            'sel': 'tour',
            'target': 'MAP',
        }

    # -- population helpers -------------------------------------------------
    def _genes(self, pop):
        keys = self.Model.FreeParameters
        return [{k: pt.Pars[k] for k in keys} for pt in pop]

    def _fitness(self, pop):
        target = self.Settings['target']
        return np.array([fitness_of(pt, target) for pt in pop])

    def _evaluate(self, genes):
        if self.Settings['parallel']:
            with Parallel(n_jobs=self.Settings['n_core'], verbose=self.Settings['verbose']) as parallel:
                res = parallel(delayed(evaluate_gene)(self.Model, g, unpack=True) for g in genes)
            return [Particle.from_json(sim) if sim is not None else None for sim, _ in res]
        return [evaluate_gene(self.Model, g)[0] for g in tqdm(genes, 'Evaluate')]

    # -- Fitter protocol -------------------------------------------------
    def initialise(self):
        self.info('Initialising')
        n = self.Settings['n_collect']

        if self.Settings['parallel']:
            with Parallel(n_jobs=self.Settings['n_core'], verbose=self.Settings['verbose']) as parallel:
                samples = parallel(delayed(sample_prior)(self.Model, unpack=True) for _ in range(n))
            pop = [Particle.from_json(sim) for sim, _ in samples]
            n_eval = sum(ne for _, ne in samples)
        else:
            pop, n_eval = list(), 0
            for _ in tqdm(range(n), 'Genesis'):
                sim, ne = sample_prior(self.Model)
                pop.append(sim)
                n_eval += ne

        self.State = StateGA(pop, self._fitness(pop))
        self._elitism(n_eval)

    def update(self, **kwargs):
        while True:
            self._a_round()
            if self._check_termination():
                break

    def terminate(self):
        pass

    def _a_round(self):
        st = self.State
        st.Generation += 1
        keys = self.Model.FreeParameters
        n = self.Settings['n_collect']

        # selection
        scored = list(zip(st.Fitness, self._genes(st.Population)))
        genes = self.Selector.select(scored, n)

        # crossover
        rd.shuffle(genes)
        for i in range(0, len(genes) - 1, 2):
            if rd.random() < self.Settings['p_cro']:
                genes[i], genes[i + 1] = self.Crossover.crossover(genes[i], genes[i + 1], keys)

        # mutation
        self.Mutator.set_scales(genes, keys)
        for i in range(len(genes)):
            if rd.random() < self.Settings['p_mut']:
                genes[i] = self.Mutator.propose(genes[i], keys)

        # evaluation; invalid proposals fall back to the incumbent best
        pop = [pt if pt is not None else st.Best for pt in self._evaluate(genes)]
        fit = self._fitness(pop)

        # elitism: never lose the best solution found so far
        if st.Best is not None and st.MaxFitness > fit.max():
            worst = int(np.argmin(fit))
            pop[worst] = st.Best
            fit[worst] = st.MaxFitness

        st.Population, st.Fitness = pop, fit
        self._elitism()

    def _elitism(self, n_eval=0):
        st = self.State
        prev = st.MaxFitness

        i = int(np.argmax(st.Fitness))
        st.Best = st.Population[i]
        st.MaxFitness = float(st.Fitness[i])

        finite = st.Fitness[np.isfinite(st.Fitness)]
        st.MeanFitness = float(logsumexp(finite) - np.log(len(finite))) if len(finite) else -np.inf

        st.Stay = st.Stay + 1 if prev >= st.MaxFitness else 0

        self.Monitor.keep(Round=st.Generation, Stay=st.Stay, Eval=n_eval,
                          MaxFitness=st.MaxFitness, MeanFitness=st.MeanFitness)
        self.Monitor.step()
        self.info(f'Round {st.Generation}, Max fitness {st.MaxFitness:.4g}, '
                  f'Mean fitness {st.MeanFitness:.4g}')

    def _check_termination(self):
        st = self.State
        if st.Generation >= self.Settings['max_round']:
            return True
        # if st.Stay >= self.Settings['max_stay']:
        #     self.info('Early terminated due to convergence')
        #     return True
        return False

    def sample_posteriors(self, n_collect=300) -> ParameterSet:
        assert self.State is not None
        self.info('Collecting the final population')
        st = self.State

        order = np.argsort(st.Fitness)[::-1]
        chosen = [st.Population[i] for i in order if np.isfinite(st.Fitness[i])]
        if not chosen:
            chosen = [st.Best]
        chosen = chosen[:n_collect]
        if len(chosen) < n_collect:
            pad = rd.choice(len(chosen), n_collect - len(chosen))
            chosen = chosen + [chosen[i] for i in pad]

        post = ParameterSet('GeneticAlg')
        for pt in chosen:
            self.Model.flatten(pt)
            post.append(pt)

        post.keep('Trace', self.Monitor.Trajectories)
        post.keep('Best', dict(st.Best.Pars))
        post.keep('MaxFitness', st.MaxFitness)
        return post


if __name__ == '__main__':
    from sims_pars.fit.toys import get_betabin

    model0 = get_betabin((4, 12))
    print('Free parameters:', model0.FreeParameters)

    alg = GeneticAlg(n_collect=200, max_round=30, parallel=True, verbose=0, sel='tour')
    alg.fit(model0)

    print('Best:', alg.State.Best.Pars)
    print(alg.sample_posteriors(200).to_df().describe())
