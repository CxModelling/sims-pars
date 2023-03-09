from sims_pars.fit.targets.base import AbsData

__author__ = 'Chu-Chang Ku'
__all__ = ['TimeSeries']


class TimeSeries(AbsData):
    def calc_distance(self, sim) -> float:
        pass

    def calc_likelihood(self, sim) -> float:
        raise AttributeError('Undefined data distibution')


