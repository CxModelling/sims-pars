from sims_pars.fit.targets import read_targets
from sims_pars.fit.base import DataModel, Particle
from sims_pars import bayes_net_from_script, sample
import scipy.stats as sts

__all__ = ['get_betabin', 'get_normal2']


class BetaBin(DataModel):
    def __init__(self, data):
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
        bn = bayes_net_from_script(scr)
        dat = read_targets({
            'x1': data[0],
            'x2': data[1]
        })

        DataModel.__init__(self, dat, bn)

        self.DataBN = bayes_net_from_script('''
        PCore Data {
            x1 ~ binom(10, p1)
            x2 ~ binom(20, p2) 
        }
        ''')

    def simulate(self, pars) -> Particle:
        return Particle(pars, sample(self.DataBN, pars))


def get_betabin(data=(7, 14)):
    return BetaBin(data)


class Normal2(DataModel):
    def __init__(self, data, n=10):
        scr = '''
PCore Normal2 {
    std ~ gamma(0.1, 0.1)
    mu1 ~ norm(0, std)
    mu2 ~ norm(0, std)
}
'''
        bn = bayes_net_from_script(scr)
        dat = read_targets({
            'x1': sts.norm(data[0], 1).rvs(n).mean(),
            'x2': sts.norm(data[1], 1).rvs(n).mean()
        })

        DataModel.__init__(self, dat, bn)
        self.N = n

    def simulate(self, pars) -> Particle:
        sim = {
            'x1': sts.norm(pars['mu1'], pars['std']).rvs(self.N).mean(),
            'x2': sts.norm(pars['mu2'], pars['std']).rvs(self.N).mean(),
        }
        return Particle(pars, sim)


def get_normal2(mu, n=10):
    return Normal2(mu, n=n)


if __name__ == '__main__':
    bb = get_betabin()

    p0 = bb.sample_prior()
    s0 = bb.simulate(p0)
    print(p0)
    print(bb.calc_distance(s0))

    n2 = get_normal2([30, -2])

    p1 = n2.sample_prior()
    print(p1)
    s1 = n2.simulate(p1)
    print(s1)
    print(n2.calc_distance(s1))
