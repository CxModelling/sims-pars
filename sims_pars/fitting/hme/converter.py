import numpy as np
from abc import ABCMeta, abstractmethod


__all__ = ['ConverterFF', 'ConverterIF', 'ConverterFI', 'ConverterII', 'get_converters']


class Converter(metaclass=ABCMeta):
    @abstractmethod
    def uniform2value(self, u):
        pass

    @abstractmethod
    def value2uniform(self, v):
        pass


class ConverterFF(Converter):
    def __init__(self, lower, upper):
        self.Lower = lower
        self.Upper = upper

    def uniform2value(self, u):
        return u * (self.Upper - self.Lower) + self.Lower

    def value2uniform(self, v):
        return (v - self.Lower) / (self.Upper - self.Lower)

    def __str__(self):
        return f'U(0, 1) <-> [{self.Lower}, {self.Upper}]'

    __repr__ = __str__


class ConverterFI(Converter):
    def __init__(self, lower):
        self.Lower = lower

    def uniform2value(self, u):
        if u >= 1:
            return np.inf
        return - np.log(1 - u) + self.Lower

    def value2uniform(self, v):
        return 1 - np.exp(self.Lower - v)

    def __str__(self):
        return f'U(0, 1) <-> [{self.Lower}, Inf]'

    __repr__ = __str__


class ConverterIF(Converter):
    def __init__(self, upper):
        self.Upper = upper

    def uniform2value(self, u):
        if u <= 0:
            return - np.inf
        return np.log(u) + self.Upper

    def value2uniform(self, v):
        return np.exp(v - self.Upper)

    def __str__(self):
        return f'U(0, 1) <-> [-Inf, {self.Upper}]'

    __repr__ = __str__


class ConverterII(Converter):
    def uniform2value(self, u):
        if u >= 1:
            return np.inf
        elif u <= 0:
            return - np.inf
        u = u - 0.5
        return - np.sign(u) * np.log(1 - 2 * np.abs(u))

    def value2uniform(self, v):
        if v > 0:
            return 1 - np.exp(-v) / 2
        else:
            return np.exp(v) / 2

    def __str__(self):
        return 'U(0, 1) <-> [-Inf, Inf]'

    __repr__ = __str__


def get_converters(doms):
    converters = list()

    for dom in doms:
        if np.isinf(dom.Lower):
            if np.isinf(dom.Upper):
                con = ConverterII()
            else:
                con = ConverterIF(dom.Upper)
        else:
            if np.isinf(dom.Upper):
                con = ConverterFI(dom.Lower)
            else:
                con = ConverterFF(dom.Lower, dom.Upper)
        converters.append(con)
    return converters


if __name__ == '__main__':
    print('\nFin-Fin')
    cff = ConverterFF(-7, 7)
    print(cff)
    vs = [cff.uniform2value(u) for u in np.linspace(0, 1, 5)]
    us = [cff.value2uniform(v) for v in vs]
    print(vs)
    print(us)

    print('\nFin-Inf')
    cfi = ConverterFI(7)
    print(cfi)
    vs = [cfi.uniform2value(u) for u in np.linspace(0, 1, 5)]
    us = [cfi.value2uniform(v) for v in vs]
    print(vs)
    print(us)

    print('\nInf-Fin')
    cif = ConverterIF(7)
    print(cif)
    vs = [cif.uniform2value(u) for u in np.linspace(0, 1, 5)]
    us = [cif.value2uniform(v) for v in vs]
    print(vs)
    print(us)

    print('\nInf-Inf')
    cii = ConverterII()
    print(cii)
    vs = [cii.uniform2value(u) for u in np.linspace(0, 1, 5)]
    us = [cii.value2uniform(v) for v in vs]
    print(vs)
    print(us)



