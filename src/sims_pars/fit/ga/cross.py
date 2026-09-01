from abc import ABCMeta, abstractmethod
import numpy.random as rd
from sims_pars.factory import get_atelier, AbsCreator

__author__ = 'Chu-Chang Ku'
__all__ = ['get_crossover']

# Crossover operators act on plain free-parameter dicts ({name: value}); the
# GA re-serves the offspring through the model afterwards.


class AbsCrossover(metaclass=ABCMeta):
    @abstractmethod
    def crossover(self, g1: dict, g2: dict, keys) -> tuple[dict, dict]:
        pass


class AverageCrossover(AbsCrossover):
    def crossover(self, g1, g2, keys):
        child = {k: (g1[k] + g2[k]) / 2 for k in keys}
        return dict(child), dict(child)


class ShuffleCrossover(AbsCrossover):
    def crossover(self, g1, g2, keys):
        c1, c2 = dict(), dict()
        for k in keys:
            if rd.random() < 0.5:
                c1[k], c2[k] = g2[k], g1[k]
            else:
                c1[k], c2[k] = g1[k], g2[k]
        return c1, c2


CrossoverCentre = get_atelier('crossover')


def get_crossover(seq):
    if not seq.endswith(')'):
        seq += '()'
    return CrossoverCentre.create(seq, append_src=False)


class CreAvg(AbsCreator):
    def create(self):
        return AverageCrossover()


CrossoverCentre.register('average', CreAvg)


class CreShu(AbsCreator):
    def create(self):
        return ShuffleCrossover()


CrossoverCentre.register('shuffle', CreShu)


if __name__ == '__main__':
    from sims_pars.fit.toys import get_betabin

    model0 = get_betabin((4, 12))
    ks = model0.FreeParameters
    g1 = {k: model0.sample_prior()[k] for k in ks}
    g2 = {k: model0.sample_prior()[k] for k in ks}

    for name in ('average', 'shuffle'):
        print(name, get_crossover(name).crossover(g1, g2, ks))
