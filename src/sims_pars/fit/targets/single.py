from sims_pars.fit.targets.base import AbsData
from abc import ABCMeta
import numpy as np

__author__ = 'Chu-Chang Ku'
__all__ = ['AbsSingleData', 'DataPointError', 'DataPointRange', 'DataDistribution']


class AbsSingleData(AbsData, metaclass=ABCMeta):
    pass


class DataPointError(AbsSingleData):
    def __init__(self, name, v, error=0.1):
        AbsSingleData.__init__(self, name, v)
        self.Error = error
        self.Std = error * v / 2 / 1.96

    def get_var_obs(self):
        return self.Std ** 2

    def calc_distance(self, sim) -> float:
        return np.power(sim[self.Name] - self.Value, 2) / self.Std

    def calc_likelihood(self, sim) -> float:
        raise AttributeError('Undefined data distribution')

    def __str__(self):
        return f'Data::{self.Name}~Error(mean={self.Value:.2g}, +/- {self.Error:.0%})'

    __repr__ = __str__


class DataPointRange(AbsSingleData):
    def __init__(self, name, m, l, u):
        AbsSingleData.__init__(self, name, m)
        self.Lower, self.Upper = l, u
        self.Range = u - l

    def get_var_obs(self):
        return (self.Range / 2 / 1.96) ** 2

    def calc_distance(self, sim) -> float:
        return np.power(sim[self.Name] - self.Value, 2) / self.Range * (2 * 1.96)

    def calc_likelihood(self, sim) -> float:
        raise AttributeError('Undefined data distribution')

    def __str__(self):
        return f'Data::{self.Name}~Range(mean={self.Value:.2g}, {self.Lower:.2g}-{self.Upper:.2g})'

    __repr__ = __str__


class DataDistribution(AbsSingleData):
    def __init__(self, name, dist):
        AbsSingleData.__init__(self, name, dist.mean())
        self.Distribution = dist

    def get_var_obs(self):
        return (self.Distribution.std()) ** 2

    def calc_distance(self, sim) -> float:
        return - self.calc_likelihood(sim)

    def calc_likelihood(self, sim) -> float:
        s = sim[self.Name]
        try:
            return self.Distribution.logpdf(s)
        except AttributeError:
            return self.Distribution.logpmf(s)

    def __str__(self):
        return f'Data::{self.Name}~Distribution(mean={self.Value:.2g}, sd={self.Distribution.std():.2g})'

    __repr__ = __str__
