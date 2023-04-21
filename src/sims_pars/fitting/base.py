from abc import ABCMeta, abstractmethod
from sims_pars.fn import evaluate_nodes, sample
from sims_pars.bayesnet import BayesianNetwork, Chromosome, bayes_net_from_json, bayes_net_from_script
from typing import Union
from collections import namedtuple

__author__ = 'Chu-Chang Ku'
__all__ = ['AbsObjective', 'AbsObjectiveBN', 'AbsObjectiveSimBased']


Domain = namedtuple('Domain', ('Name', 'Type', 'Lower', 'Upper'))


class AbsObjective(metaclass=ABCMeta):
    def __init__(self, exo=None):
        self.ExoParameters = dict(exo) if exo is not None else dict()
        self.FreeParameters = dict()

    @property
    @abstractmethod
    def Domain(self):
        pass

    @abstractmethod
    def serve(self, p: dict):
        raise AttributeError('Unknown parameter definition')

    def serve_from_json(self, js: dict):
        p = self.serve(js['Locus'])
        p.LogPrior, p.LogLikelihood = js['LogPrior'], js['LogLikelihood']
        return p

    @abstractmethod
    def sample_prior(self):
        pass

    @abstractmethod
    def evaluate_prior(self, pars: Chromosome):
        pass

    @abstractmethod
    def calc_likelihood(self, pars: Chromosome):
        pass

    def evaluate(self, pars: Chromosome) -> float:
        # prevent re-evaluation
        if not pars.is_likelihood_evaluated():
            pars.LogLikelihood = self.calc_likelihood(pars)
        return pars.LogLikelihood

    def print(self):
        print('Free parameters: {}'.format(', '.join(self.FreeParameters)))
        print('Exogenous variables: {}'.format(', '.join(self.ExoParameters)))


class AbsObjectiveBN(AbsObjective, metaclass=ABCMeta):
    def __init__(self, bn: Union[BayesianNetwork, str, dict], exo=None):
        AbsObjective.__init__(self, exo)
        if isinstance(bn, str):
            bn = bayes_net_from_script(bn)
        elif isinstance(bn, dict):
            bn = bayes_net_from_json(bn)

        self.FreeParameters = [node for node in bn.Order if bn.is_rv(node) and node not in self.ExoParameters]
        self.BayesianNetwork = bn

        # Exclude non-float
        # todo
        p0 = self.sample_prior()
        pfs = [k for k, v in dict(p0).items() if isinstance(v, float)]
        self.FreeParameters = [node for node in self.FreeParameters if node in pfs]

    def serve(self, p: dict):
        p = dict(p)
        p.update(self.ExoParameters)
        pars = Chromosome(sample(self.BayesianNetwork, p))
        self.evaluate_prior(pars)
        return pars

    @property
    def Domain(self):
        p = self.sample_prior()
        res = []
        for node in self.FreeParameters:
            d = self.BayesianNetwork[node].get_distribution(p)
            res.append(Domain(Name=node, Type=d.Type, Upper=d.Upper, Lower=d.Lower))
        return res

    def sample_prior(self):
        pars = sample(self.BayesianNetwork, self.ExoParameters)
        pars.update(self.ExoParameters)
        pars = Chromosome(pars)
        self.evaluate_prior(pars)
        return pars

    def evaluate_prior(self, p: Chromosome):
        if not p.is_prior_evaluated():
            p.LogPrior = evaluate_nodes(self.BayesianNetwork, p)
        return p.LogPrior

    @abstractmethod
    def calc_likelihood(self, pars: Chromosome):
        pass

    def print(self):
        print('Model: {}'.format(self.BayesianNetwork.Name))
        AbsObjective.print(self)


class AbsObjectiveSimBased(AbsObjectiveBN, metaclass=ABCMeta):
    def calc_likelihood(self, pars: Chromosome):
        sim = self.simulate(pars)
        if sim is None:
            raise ValueError('Invalid simulation run')
        return self.link_likelihood(sim)

    @abstractmethod
    def simulate(self, pars):
        pass

    @abstractmethod
    def link_likelihood(self, sim):
        pass


if __name__ == '__main__':
    from sims_pars import parse_distribution


    class BetaBinSimBased(AbsObjectiveSimBased):
        def simulate(self, pars):
            sim = {
                'x1': pars['x1'],
                'x2': pars['x2']
            }
            return sim

        def link_likelihood(self, sim):
            return -((sim['x1'] - 5) ** 2 + (sim['x2'] - 10) ** 2)


    class BetaBinBN(AbsObjectiveBN):
        def calc_likelihood(self, pars):
            pars = dict(pars)
            pars.update(self.ExoParameters)
            x1 = parse_distribution('binom(10, p1)', loc=pars).sample(1)
            x2 = parse_distribution('binom(n2, p2)', loc=pars).sample(1)

            return -((x1 - 5) ** 2 + (x2 - 10) ** 2)


    scr = '''
    PCore BetaBin {
        al = 1
        be = 1

        p1 ~ beta(al, be)
        p2 ~ beta(al, be)

        x1 ~ binom(10, p1)
        x2 ~ binom(n2, p2) 
    }
    '''

    model0 = BetaBinSimBased(scr, exo={'n2': 20})
    model0.print()

    print('Domain:')
    for do in model0.Domain:
        print('\t', do)

    p1 = model0.sample_prior()
    print("Parameters: ", p1)
    print("Likelihood: ", model0.evaluate(p1))
    print('\n')

    model1 = BetaBinBN(scr, exo={'n2': 20})
    model1.print()

    print('Domain:')
    for do in model1.Domain:
        print('\t', do)

    p1 = model1.sample_prior()
    print("Parameters: ", p1)
    print("Likelihood: ", model1.evaluate(p1))
