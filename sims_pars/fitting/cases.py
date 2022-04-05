from bayesnet import Chromosome
from sims_pars.fitting.base import AbsObjectiveBN
from sims_pars.simulation import get_all_fixed_sc
import scipy.stats as sts

__all__ = ['BetaBin', 'NormalTwo']


class BetaBin(AbsObjectiveBN):
    def __init__(self, data=(7, 14)):
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
        AbsObjectiveBN.__init__(self, sc)
        self.Data = list(data)

    def simulate(self, pars):
        return {
            'x1': pars['x1'],
            'x2': pars['x2']
        }

    def link_likelihood(self, sim):
        return -((sim['x1'] - self.Data[0]) ** 2 + (sim['x2'] - self.Data[1]) ** 2)

    def calc_likelihood(self, pars: Chromosome):
        pass



class NormalTwo(AbsObjectiveBN):
    def __init__(self, mu, n=10):
        sc = get_all_fixed_sc('''
        PCore Normal2 {
            mu1 ~ norm(0, 1)
            mu2 ~ norm(0, 1)
        }
        ''')
        AbsObjectiveBN.__init__(self, sc)
        self.Mu = mu
        self.N = n
        self.X1 = sts.norm(self.Mu[0], 1).rvs(n)
        self.X2 = sts.norm(self.Mu[1], 1).rvs(n)

    def simulate(self, pars):
        return {
            'mu1': pars['mu1'],
            'mu2': pars['mu2']
        }

    def link_likelihood(self, sim):
        return sts.norm.logpdf(self.X1, sim['mu1'], 1).sum() + sts.norm.logpdf(self.X2, sim['mu2'], 1).sum()

    def calc_likelihood(self, pars: Chromosome):
        pass



if __name__ == '__main__':
    bb = BetaBin()

    p0 = bb.sample_prior()
    print(p0)
    print(bb.evaluate(p0))

    n2 = NormalTwo([30, -2])

    p1 = n2.sample_prior()
    print(p1)
    print(n2.evaluate(p1))
