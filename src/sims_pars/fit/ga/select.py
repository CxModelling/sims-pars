from pydantic import PositiveInt
from abc import ABCMeta, abstractmethod
import numpy as np
import numpy.random as rd
from scipy.special import logsumexp
from sims_pars.factory import get_atelier, AbsCreator

__author__ = 'Chu-Chang Ku'
__all__ = ['get_selector']

# Selectors consume a list of (fitness, gene) pairs -- fitness to be maximised,
# gene a plain free-parameter dict -- and return a fresh list of genes.


class AbsSelector(metaclass=ABCMeta):
    @abstractmethod
    def select(self, scored, n) -> list:
        pass

    @staticmethod
    def _eligible(scored):
        return [(f, g) for f, g in scored if np.isfinite(f)]


class TourSelection(AbsSelector):
    def __init__(self, k):
        self.K = k

    def select(self, scored, n):
        elig = self._eligible(scored)
        assert len(elig) >= self.K, 'not enough finite-fitness genes for a tournament'

        out = list()
        while len(out) < n:
            cand = [elig[i] for i in rd.choice(len(elig), self.K, replace=False)]
            out.append(dict(max(cand, key=lambda t: t[0])[1]))
        return out


class ImpSelection(AbsSelector):
    def select(self, scored, n):
        elig = self._eligible(scored)
        assert len(elig) > 5, 'not enough finite-fitness genes for importance sampling'

        wts = np.array([f for f, _ in elig])
        wts -= logsumexp(wts)
        idx = rd.choice(len(elig), n, replace=True, p=np.exp(wts))
        return [dict(elig[i][1]) for i in idx]


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
    from sims_pars.fit.toys import get_betabin
    from sims_pars.fit.ga.util import sample_prior, fitness_of

    model0 = get_betabin((4, 12))
    ks = model0.FreeParameters
    pop = [sample_prior(model0)[0] for _ in range(30)]
    scored = [(fitness_of(pt, 'MAP'), {k: pt.Pars[k] for k in ks}) for pt in pop]

    print('tour   ->', len(get_selector('tour(5)').select(scored, 30)))
    print('import ->', len(get_selector('importance').select(scored, 30)))
