from abc import ABCMeta, abstractmethod
import numpy.random as rd
from sims_pars.factory import get_atelier, AbsCreator
from sims_pars.fitting.base import AbsObjective

__author__ = 'Chu-Chang Ku'
__all__ = ['get_crossover']


class AbsCrossover(metaclass=ABCMeta):
    @abstractmethod
    def crossover(self, p1, p2, obj: AbsObjective):
        pass


class AverageCrossover(AbsCrossover):
    def crossover(self, p1, p2, obj: AbsObjective):
        locus = dict()
        for node in obj.Domain:
            locus[node.Name] = (p1[node.Name] + p2[node.Name]) / 2

        return [locus, dict(locus)]


class ShuffleCrossover(AbsCrossover):
    def crossover(self, p1, p2, obj: AbsObjective):
        locus1, locus2 = dict(), dict()
        for node in obj.Domain:
            node = node.Name
            if rd.random() < 0.5:
                locus1[node] = p2[node]
                locus2[node] = p1[node]
            else:
                locus1[node] = p1[node]
                locus2[node] = p2[node]


        return [locus1, locus2]


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
    from sims_pars.fitting.util import draw, serve_and_evaluate
    from sims_pars.fitting.cases import BetaBin

    model0 = BetaBin()

    print('Free parameters: ', model0.FreeParameters)
    for p in model0.Domain:
        print(p)

    ps = [draw(model0) for _ in range(2)]
    ps = [p for p, i in ps]

    print('p1', ps[0])
    print('p2', ps[1])

    print('Average crossover')
    cro0 = get_crossover('average')

    p1, p2 = cro0.crossover(*ps, model0)

    p1 = serve_and_evaluate(model0, p1)
    p2 = serve_and_evaluate(model0, p2)

    print('avg: ', p1)

    print('Shuffle crossover')
    cro0 = get_crossover('shuffle')

    p1, p2 = cro0.crossover(*ps, model0)

    p1 = serve_and_evaluate(model0, p1)
    p2 = serve_and_evaluate(model0, p2)

    print('shuffle: ', p1)
