import scipy.stats as sts
from scipy.interpolate import interp1d
import numpy as np
from abc import ABCMeta, abstractmethod
import numpy.random as rd
from sims_pars.factory import get_atelier, AbsCreator
import typing
import pydantic.types as ptp


__author__ = 'TimeWz667'
__all__ = ['AbsDistribution',
           'SpDouble',
           'SpInteger',
           'DistributionCentre',
           'parse_distribution',
           'complete_function',
           'CategoricalRV']


class AbsDistribution(metaclass=ABCMeta):
    def __init__(self):
        self.source = None
        self.json = None

    def __call__(self, **kwargs):
        return self.sample(1, **kwargs)

    @property
    @abstractmethod
    def Interval(self):
        pass

    @property
    def Upper(self):
        return self.Interval[1]

    @property
    def Lower(self):
        return self.Interval[0]

    @property
    @abstractmethod
    def Type(self):
        pass

    @abstractmethod
    def sample(self, n=1, **kwargs):
        pass

    @abstractmethod
    def logpdf(self, v):
        pass

    @abstractmethod
    def mean(self):
        pass

    @abstractmethod
    def std(self):
        pass

    def to_json(self):
        if self.json:
            return self.json
        else:
            raise AttributeError('Undefined JSON format')

    def __repr__(self):
        if self.source:
            return repr(self.source)
        else:
            return id(self)

    __str__ = __repr__


class SpDouble(AbsDistribution):
    def __init__(self, dist):
        AbsDistribution.__init__(self)
        self.Dist = dist

    def sample(self, n=1, **kwargs):
        if n == 1:
            return self.Dist.rvs()
        return self.Dist.rvs(n)

    @property
    def Interval(self):
        return self.Dist.interval(1)

    @property
    def Type(self):
        return 'Double'

    def logpdf(self, v):
        return self.Dist.logpdf(v)

    def mean(self):
        return self.Dist.mean()

    def std(self):
        return self.Dist.std()


class SpInteger(AbsDistribution):
    def __init__(self, dist):
        AbsDistribution.__init__(self)
        self.Dist = dist

    def sample(self, n=1, **kwargs):
        if n == 1:
            return round(self.Dist.rvs())
        return np.round(self.Dist.rvs(n))

    @property
    def Interval(self):
        inter = self.Dist.interval(1)
        return inter[0]+1, inter[1]

    @property
    def Type(self):
        return 'Integer'

    def logpdf(self, v):
        return self.Dist.logpmf(v)

    def mean(self):
        return self.Dist.mean()

    def std(self):
        return self.Dist.std()


class Const(AbsDistribution):
    def __init__(self, k):
        """
        Distribution function always draws a constant value
        :param k: value
        """
        AbsDistribution.__init__(self)
        self.K = k

    @property
    def Interval(self):
        return [self.K, self.K]

    @property
    def Upper(self):
        return self.K

    @property
    def Lower(self):
        return self.K

    @property
    def Type(self):
        return type(self.K)

    def sample(self, n=1, **kwargs):
        if n > 1:
            return np.full(n, self.K)
        return self.K

    def logpdf(self, v):
        return 0 if v == self.K else float('inf')

    def mean(self):
        return self.K

    def std(self):
        return 0


class CategoricalRV(AbsDistribution):
    """
    Generate Categorical data with respect to a specific probability distribution.
    """

    def __init__(self, kv):
        """

        :param kv: a dictionary with keys of category names and values of probabilities.
        """
        AbsDistribution.__init__(self)
        self.kv = kv
        self.cat = [k for k in kv.keys()]
        self.p = np.array(list(kv.values()))
        self.p = self.p / self.p.sum()

    @property
    def Dist(self):
        return "Cat([{}])".format(','.join(self.cat))

    @property
    def Interval(self):
        return None, None

    @property
    def Type(self):
        return 'Category'

    def logpdf(self, v):
        try:
            return np.array([x*np.log(self.kv[k]) for k, x in v.items()]).sum()
        except AttributeError:
            return np.log(self.kv[v])

    def sample(self, n=1, **kwargs):
        sam = rd.choice(self.cat, n, p=self.p)
        if n == 1:
            return sam[0]
        else:
            return sam

    def mean(self):
        return 0

    def std(self):
        return 0


class EmpiricalRV(AbsDistribution):
    def __init__(self, x, y):
        x, y = np.array(x), np.array(y)
        self.X = x
        self.Y = y
        self.__int = (x.min(), x.max())
        AbsDistribution.__init__(self)
        self.Fn = interp1d(y.cumsum()/y.sum(), x, bounds_error=False, fill_value=(x.min(), x.max()))
        self.Logpdf = interp1d(x, y, bounds_error=False, fill_value=0)

    @property
    def Interval(self):
        return self.__int

    @property
    def Type(self):
        return 'Double'

    def logpdf(self, v):
        return self.Logpdf(v)

    def sample(self, n=1, **kwargs):
        return self.Fn(rd.random(n))

    def mean(self):
        return np.mean(self.X*self.Fn(self.X))

    def std(self):
        return np.mean(self.X*self.X*self.Fn(self.X))


DistributionCentre = get_atelier('Distributions')


class CreConst(AbsCreator):
    k: float = 0

    def create(self):
        return Const(self.k)


DistributionCentre.register('k', CreConst)


class CreGamma(AbsCreator):
    shape: ptp.PositiveFloat = 1.0
    rate: ptp.PositiveFloat = 1.0

    def create(self):
        return SpDouble(sts.gamma(a=self.shape, scale=1/self.rate))


DistributionCentre.register('gamma', CreGamma)


class CreExp(AbsCreator):
    rate: ptp.PositiveFloat = 1.0

    def create(self):
        return SpDouble(sts.expon(scale=1/self.rate))


DistributionCentre.register('exp', CreExp)


class CreLogNorm(AbsCreator):
    meanlog: ptp.NonNegativeFloat = 0
    sdlog: ptp.PositiveFloat = 1

    def create(self):
        return SpDouble(sts.lognorm(s=np.exp(self.sdlog), scale=np.exp(np.exp(self.meanlog))))


DistributionCentre.register('lnorm', CreLogNorm)


class CreUniform(AbsCreator):
    min: float = 0
    max: float = 1

    def create(self):
        return SpDouble(sts.uniform(self.min, self.max-self.min))


DistributionCentre.register('unif', CreUniform)


class CreChi2(AbsCreator):
    df: ptp.PositiveFloat = 1.0

    def create(self):
        return SpDouble(sts.chi2(self.df))


DistributionCentre.register('chisq', CreChi2)


class CreBeta(AbsCreator):
    shape1: ptp.PositiveFloat = 1.0
    shape2: ptp.PositiveFloat = 1.0

    def create(self):
        return SpDouble(sts.beta(self.shape1, self.shape2))


DistributionCentre.register('beta', CreBeta)


class CreInvGamma(AbsCreator):
    a: ptp.confloat(gt=2) = 2.0
    rate: ptp.PositiveFloat = 1.0

    def create(self):
        return SpDouble(sts.invgamma(a=self.a, scale=1/self.rate))


DistributionCentre.register('invgamma', CreInvGamma)


class CreNorm(AbsCreator):
    mean: float = 0
    sd: ptp.PositiveFloat = 1.0

    def create(self):
        return SpDouble(sts.norm(loc=self.mean, scale=self.sd))


DistributionCentre.register('norm', CreNorm)


class CreTriangle(AbsCreator):
    a: float = 0
    m: float = 0.5
    b: float = 1

    def create(self):
        x = [self.a, self.m, self.b]
        x.sort()
        a, m, b = x
        return SpDouble(sts.triang(loc=a, scale=b - a, c=(m - a) / (b - a)))


DistributionCentre.register('triangle', CreTriangle)


class CreBinom(AbsCreator):
    size: ptp.PositiveInt = 1
    prob: ptp.confloat(ge=0, le=1) = 0.5

    def create(self):
        return SpInteger(sts.binom(n=self.size, p=self.prob))


DistributionCentre.register('binom', CreBinom)


class CrePois(AbsCreator):
    lam: ptp.PositiveFloat = 1

    def create(self):
        return SpInteger(sts.poisson(mu=self.lam))


DistributionCentre.register('pois', CrePois)


class CreCat(AbsCreator):
    kv: typing.Dict[str, float]

    def create(self):
        return CategoricalRV(self.kv)


DistributionCentre.register('cat', CreCat)


def parse_distribution(di, loc=None, append_src=True, to_complete=True):
    return DistributionCentre.create(di, loc=loc, append_src=append_src, to_complete=to_complete)


def complete_function(di):
    return DistributionCentre.complete_def(di)


if __name__ == '__main__':
    dists = [
        'k(1)',
        'gamma(0.01, 1)',
        'exp(0.01)',
        'lnorm(0.5, 1)',
        'unif(0, 1)',
        'chisq(20)',
        'triangle(2, 3, 5)',
        'binom(size=4, prob=0.5)'
    ]
    dists = [parse_distribution(di) for di in dists]
    for di in dists:
        print(di)
        print(di.to_json())
        print(di.mean())

    dist_cat = parse_distribution('cat({"M": 411,"O": kk,"Y": 52})', loc={'kk': 3500})
    from collections import Counter
    print(dist_cat.to_json())
    print(Counter(dist_cat.sample(10000)))
