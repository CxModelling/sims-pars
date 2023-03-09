from sims_pars.fit.targets.base import *
from sims_pars.fit.targets.single import *
from sims_pars.fit.targets.multi import *
from sims_pars import parse_distribution

__all__ = ['read_targets', 'AbsSingleData']


def read_targets(src, error=0.1):
    parsed = dict()
    for k, v in src.items():
        if isinstance(v, float) or isinstance(v, int):
            pt = DataPointError(k, v, error)

        elif isinstance(v, dict):
            if 'm' not in v:
                raise ValueError('Unknown data format')
            if 'l' in v and 'u' in v:
                m, l, u = v['m'], v['l'], v['u']
                assert l < m < u
                pt = DataPointRange(k, m, l, u)
            else:
                try:
                    e = v['error']
                except KeyError:
                    e = error
                pt = DataPointError(k, v['m'], e)
        elif isinstance(v, str):
            pt = DataDistribution(k, parse_distribution(v))
        elif isinstance(v, AbsData):
            pt = v
        else:
            raise ValueError('Unknown data format')

        parsed[k] = pt
    return parsed


if __name__ == '__main__':

    ds = {
        'T1': 0.5,
        'T2': {'m': 0.5, 'l': 0.2, 'u': 0.8},
        'T3': 'unif(0.2, 0.8)'
    }

    targets = read_targets(ds)
    print(targets)
