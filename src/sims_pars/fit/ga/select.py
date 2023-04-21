from pydantic.types import PositiveInt
from abc import ABCMeta, abstractmethod
import numpy as np
import numpy.random as rd
from scipy.special import logsumexp
from sims_pars.factory import get_atelier, AbsCreator
from sims_pars.fitting.base import AbsObjective

__all__ = ['get_selector']
__author__ = 'Chu-Chang Ku'


class AbsSelector(metaclass=ABCMeta):
    @abstractmethod
    def select(self, ps, obj: AbsObjective, fitness='MAP'):
        pass


class TourSelection(AbsSelector):
    def __init__(self, k):
        self.K = k

    def select(self, ps, obj: AbsObjective, fitness='MAP'):
        n_pop = len(ps)
        eligible = [p for p in ps if np.isfinite(p.LogLikelihood)]

        assert len(eligible) > self.K

        sel = list()
        while len(sel) < n_pop:
            candidates = [eligible[i] for i in rd.choice(len(eligible), self.K, replace=False)]
            if fitness == 'MAP':
                winner = max(candidates, key=lambda p: p.LogPosterior)
            else:
                winner = max(candidates, key=lambda p: p.LogLikelihood)

            sel.append(winner.clone())
        return sel


class ImpSelection(AbsSelector):
    def select(self, ps, obj: AbsObjective, fitness='MAP'):
        n_pop = len(ps)
        eligible = [p for p in ps if np.isfinite(p.LogLikelihood)]

        assert len(eligible) > 5

        if fitness == 'MAP':
            wts = np.array([p.LogPosterior for p in eligible])
        else:
            wts = np.array([p.LogLikelihood for p in eligible])

        wts -= logsumexp(wts)

        sel = rd.choice(len(eligible), n_pop, replace=True, p=np.exp(wts))
        sel = [eligible[i].clone() for i in sel]
        return sel


SelectCentre = get_atelier('selector')


def get_selector(seq):
    if not seq.endswith(')'):
        seq += '()'
    return SelectCentre.create(seq, append_src=False)


class CreTour(AbsCreator):
    K: PositiveInt = 3

    def create(self):
        return TourSelection(self.K)


SelectCentre.register('tour', CreTour)


class CreImp(AbsCreator):
    def create(self):
        return ImpSelection()


SelectCentre.register('importance', CreImp)


if __name__ == '__main__':
    from sims_pars.fitting.util import draw
    from sims_pars.fitting.cases import BetaBin

    model0 = BetaBin()

    print('Free parameters: ', model0.FreeParameters)
    for par in model0.Domain:
        print(par)

    ps0 = [draw(model0) for _ in range(30)]
    ps0 = [p for p, _ in ps0]
    print(np.mean([p.LogLikelihood for p in ps0]))

    sel0 = get_selector('tour(5)')
    ps1 = sel0.select(ps0, model0)
    print(np.mean([p.LogLikelihood for p in ps1]))

    sel0 = get_selector('importance')
    ps2 = sel0.select(ps0, model0)
    print(np.mean([p.LogLikelihood for p in ps2]))
