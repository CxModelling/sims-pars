from abc import ABCMeta, abstractmethod
import numpy as np
import numpy.random as rd
import scipy.stats as sts
from sims_pars.factory import get_atelier, AbsCreator

__author__ = 'Chu-Chang Ku'
__all__ = ['get_mutator']

# Mutators act on plain free-parameter dicts ({name: value}); proposals that
# leave the prior support are rejected by the GA when it re-serves them.


class AbsMutator(metaclass=ABCMeta):
    def __init__(self):
        self.Scales = dict()

    @abstractmethod
    def set_scales(self, genes, keys):
        pass

    @abstractmethod
    def propose(self, gene, keys) -> dict:
        pass


class RwMutator(AbsMutator):
    """Gaussian random-walk kernel with a Silverman-style bandwidth per gene."""

    def set_scales(self, genes, keys):
        for k in keys:
            x = np.array([g[k] for g in genes], dtype=float)
            hi = x.std()
            lo = min(sts.iqr(x), sts.iqr(x) / 1.34)
            if not lo:
                lo = hi if hi else 0.1
            self.Scales[k] = 0.9 * lo * np.power(len(x), -0.2)

    def propose(self, gene, keys):
        return {k: rd.normal(gene[k], self.Scales.get(k, 0.1)) for k in keys}


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
    from sims_pars.fit.toys import get_betabin

    model0 = get_betabin((4, 12))
    ks = model0.FreeParameters
    pool = [{k: model0.sample_prior()[k] for k in ks} for _ in range(30)]

    mut0 = get_mutator('rw')
    mut0.set_scales(pool, ks)
    print('Scales:', mut0.Scales)
    print('proposal:', mut0.propose(pool[0], ks))
