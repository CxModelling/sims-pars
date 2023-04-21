from abc import ABCMeta, abstractmethod
import numpy as np
import numpy.random as rd
import scipy.stats as sts
from sims_pars.factory import get_atelier, AbsCreator
from sims_pars.fitting.base import AbsObjective

__all__ = ['get_mutator']
__author__ = 'Chu-Chang Ku'


class AbsMutator(metaclass=ABCMeta):
    def __init__(self):
        self.Scales = dict()

    @abstractmethod
    def set_scales(self, ps, obj: AbsObjective):
        pass

    @abstractmethod
    def propose(self, p, obj: AbsObjective):
        pass


class RwMutator(AbsMutator):
    def __init__(self):
        AbsMutator.__init__(self)
        self.Eps = np.finfo(float).eps

    def set_scales(self, ps, obj: AbsObjective):
        for node in obj.Domain:
            node = node.Name

            vs = [p[node] for p in ps]

            x = np.array(vs)
            hi, lo = x.std(), min(sts.iqr(x), sts.iqr(x) / 1.34)
            if not lo:
                lo = hi if hi else .1

            self.Scales[node] = 0.9 * lo * np.power(len(x), -0.2)

    def propose(self, p, obj: AbsObjective):
        locus = dict()

        for node in obj.Domain:
            name = node.Name
            v1 = rd.normal(p[name], self.Scales[name])
            locus[name]= max(min(v1, node.Upper - self.Eps), node.Lower + self.Eps)

        return locus


MutatorCentre = get_atelier('mutator')


def get_mutator(seq):
    if not seq.endswith(')'):
        seq += '()'
    return MutatorCentre.create(seq, append_src=False)


class CreRw(AbsCreator):
    def create(self):
        return RwMutator()


MutatorCentre.register('rw', CreRw)


if __name__ == '__main__':
    from sims_pars.fitting.util import draw, serve_and_evaluate
    from sims_pars.fitting.cases import BetaBin

    model0 = BetaBin()

    print('Free parameters: ', model0.FreeParameters)
    for p in model0.Domain:
        print(p)

    ps0 = [draw(model0) for _ in range(30)]
    ps0 = [p for p, _ in ps0]

    p0, _ = draw(model0)
    print('p0:', p0)

    print('Random walk mutator')
    mut0 = get_mutator('rw')
    mut0.set_scales(ps0, model0)
    print('Scales: ', mut0.Scales)

    p1 = mut0.propose(p0, model0)
    p1 = serve_and_evaluate(model0, p1)

    print('rw: ', p1)

