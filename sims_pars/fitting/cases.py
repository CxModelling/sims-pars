from sims_pars.fitting import AbsTarget
from sims_pars.simulation import get_all_fixed_sc


__all__ = ['BetaBin']


class BetaBin(AbsTarget):
    def __init__(self):
        scr = '''
        PCore BetaBin {
            al = 1
            be = 1

            p1 ~ beta(al, be)
            p2 ~ beta(al, be)

            x1 ~ binom(10, p1)
            x2 ~ binom(20, p2) 
        }
        '''
        sc = get_all_fixed_sc(scr)
        AbsTarget.__init__(self, sc)

    def calc_likelihood(self, pars):
        return -((pars['x1'] - 5) ** 2 + (pars['x2'] - 10) ** 2)


if __name__ == '__main__':
    bb = BetaBin()

    p0 = bb.sample_prior()
    print(p0)

    print(bb.evaluate(p0))
