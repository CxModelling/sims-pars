from abc import ABCMeta, abstractmethod
from functools import lru_cache

__author__ = 'Chu-Chang Ku'
__all__ = ['AbsData']


class AbsData(metaclass=ABCMeta):
    def __init__(self, name, value):
        self.Name = name
        self.Value = value

    @abstractmethod
    def get_var_obs(self):
        pass

    @lru_cache
    def ev(self):
        return self.Value, self.get_var_obs()

    @abstractmethod
    def calc_distance(self, sim) -> float:
        pass

    @abstractmethod
    def calc_likelihood(self, sim) -> float:
        pass
