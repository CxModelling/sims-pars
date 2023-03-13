import numpy as np
from abc import ABCMeta, abstractmethod
from sims_pars.fit.base import Domain

__all__ = ['ConverterFF', 'ConverterIF', 'ConverterFI', 'ConverterII', 'get_converters']


class Converter(metaclass=ABCMeta):
    @abstractmethod
    def uniform2value(self, u):
        pass

    @abstractmethod
    def value2uniform(self, v):
        pass


class ConverterFF(Converter):
    def __init__(self, dom: Domain):
        self.Lower = dom.Lower
        self.Upper = dom.Upper

    def uniform2value(self, u):
        return u * (self.Upper - self.Lower) + self.Lower

    def value2uniform(self, v):
        return (v - self.Lower) / (self.Upper - self.Lower)

    def __str__(self):
        return f'U(0, 1) <-> [{self.Lower}, {self.Upper}]'

    __repr__ = __str__


class ConverterFI(Converter):
    def __init__(self, dom: Domain):
        self.Lower = dom.Lower
        self.Scale = dom.Scale

    def uniform2value(self, u):
        if u >= 1:
            return np.inf
        return - np.log(1 - u) * self.Scale + self.Lower

    def value2uniform(self, v):
        return 1 - np.exp((self.Lower - v) / self.Scale)

    def __str__(self):
        return f'U(0, 1) <-> [{self.Lower}, Inf]'

    __repr__ = __str__


class ConverterIF(Converter):
    def __init__(self, dom: Domain):
        self.Upper = dom.Upper
        self.Scale = dom.Scale

    def uniform2value(self, u):
        if u <= 0:
            return - np.inf
        return np.log(u) * self.Scale + self.Upper

    def value2uniform(self, v):
        return np.exp((v - self.Upper) / self.Scale)

    def __str__(self):
        return f'U(0, 1) <-> [-Inf, {self.Upper}]'

    __repr__ = __str__


class ConverterII(Converter):
    def __init__(self, dom: Domain):
        self.Scale = dom.Scale
        self.Location = dom.Location

    def uniform2value(self, u):
        if u >= 1:
            return np.inf
        elif u <= 0:
            return - np.inf
        u = u - 0.5
        return - np.sign(u) * np.log(1 - 2 * np.abs(u)) * self.Scale + self.Location

    def value2uniform(self, v):
        v = v - self.Location
        if v > 0:
            return 1 - np.exp(-v / self.Scale) / 2
        else:
            return np.exp(v / self.Scale) / 2

    def __str__(self):
        return 'U(0, 1) <-> [-Inf, Inf]'

    __repr__ = __str__


def get_converter(dom: Domain):
    if np.isinf(dom.Lower):
        if np.isinf(dom.Upper):
            con = ConverterII(dom)
        else:
            con = ConverterIF(dom)
    else:
        if np.isinf(dom.Upper):
            con = ConverterFI(dom)
        else:
            con = ConverterFF(dom)
    return con


if __name__ == '__main__':
    dom = Domain('Fin-Fin', 'float', -7, 7)
    print(dom)
    cff = get_converter(dom)
    print(cff)
    vs = [cff.uniform2value(u) for u in np.linspace(0, 1, 5)]
    us = [cff.value2uniform(v) for v in vs]
    print(vs)
    print(us)

    dom = Domain('Fin-Inf', 'float', lower=-7)
    print(dom)
    cfi = get_converter(dom)
    print(cfi)
    vs = [cfi.uniform2value(u) for u in np.linspace(0, 1, 5)]
    us = [cfi.value2uniform(v) for v in vs]
    print(vs)
    print(us)

    dom = Domain('Fin-Inf', 'float', lower=-7, scale=30)
    print(dom)
    cfi = get_converter(dom)
    print(cfi)
    vs = [cfi.uniform2value(u) for u in np.linspace(0, 1, 5)]
    us = [cfi.value2uniform(v) for v in vs]
    print(vs)
    print(us)

    dom = Domain('Inf-Fin', 'float', scale=20)
    print(dom)
    cif = get_converter(dom)
    print(cif)
    vs = [cif.uniform2value(u) for u in np.linspace(0, 1, 5)]
    us = [cif.value2uniform(v) for v in vs]
    print(vs)
    print(us)

    dom = Domain('Inf-Inf', 'float', loc=7, scale=20)
    print(dom)
    cii = get_converter(dom)
    print(cii)
    vs = [cii.uniform2value(u) for u in np.linspace(0, 1, 5)]
    us = [cii.value2uniform(v) for v in vs]
    print(vs)
    print(us)

    dom = Domain('Inf-Inf', 'float', loc=7, scale=0.1)
    print(dom)
    cii = get_converter(dom)
    print(cii)
    vs = [cii.uniform2value(u) for u in np.linspace(0, 1, 5)]
    us = [cii.value2uniform(v) for v in vs]
    print(vs)
    print(us)

