
from abc import ABCMeta, abstractmethod
from sims_pars.fn import evaluate_nodes, sample
from sims_pars.simulation.fn import find_free_parameters
from sims_pars.simulation import SimulationCore
from sims_pars.bayesnet import BayesianNetwork, Chromosome

__author__ = 'Chu-Chang Ku'
__all__ = ['AbsObjective', 'AbsObjectiveBN', 'AbsObjectiveSC']


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
    def __init__(self, bn: BayesianNetwork, exo=None):
        AbsObjective.__init__(self, exo)
        self.FreeParameters = [node for node in bn.Order if bn.is_rv(node)]
        self.BayesianNetwork = bn

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
            res.append({'Name': node, 'Type': d.Type, 'Upper': d.Upper, 'Lower': d.Lower})
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


class AbsObjectiveSC(AbsObjective, metaclass=ABCMeta):
    def __init__(self, sm: SimulationCore, exo=None):
        AbsObjective.__init__(self, exo)
        self.FreeParameters = find_free_parameters(sm, exo)
        self.SimulationCore = sm

    def serve(self, p):
        p = dict(p)
        p.update(self.ExoParameters)
        pars = self.SimulationCore.generate(self.SimulationCore.Name, p)
        self.evaluate_prior(pars)
        return pars

    @property
    def Domain(self):
        p = self.sample_prior()

        res = []
        for node in self.FreeParameters:
            d = self.SimulationCore.BN[node].get_distribution(p)
            res.append({'Name': node, 'Type': d.Type, 'Upper': d.Upper, 'Lower': d.Lower})
        return res

    def sample_prior(self):
        return self.SimulationCore.generate(exo=self.ExoParameters)

    def evaluate_prior(self, p: Chromosome):
        if not p.is_prior_evaluated():
            p.LogPrior = evaluate_nodes(self.SimulationCore.BN, p)
        return p.LogPrior

    @abstractmethod
    def simulate(self, pars):
        pass

    @abstractmethod
    def link_likelihood(self, sim):
        pass

    def calc_likelihood(self, pars: Chromosome):
        sim = self.simulate(pars)
        if sim is None:
            raise ValueError('Invalid simulation run')
        return self.link_likelihood(sim)

    def print(self):
        print('Model: {}'.format(self.SimulationCore.Name))
        AbsObjective.print(self)


if __name__ == '__main__':
    from sims_pars import parse_distribution
    from sims_pars.simulation import get_all_fixed_sc
    from sims_pars.bayesnet import bayes_net_from_script


    class BetaBinSC(AbsObjectiveSC):
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

    sc0 = get_all_fixed_sc(scr)

    sc0.deep_print()

    p0 = sc0.generate(exo={'n2': 20})
    print(p0)

    model0 = BetaBinSC(sc0, exo={'n2': 20})
    model0.print()

    print('Domain:')
    for do in model0.Domain:
        print('\t', do)

    p1 = model0.sample_prior()
    print("Parameters: ", p1)
    print("Likelihood: ", model0.evaluate(p1))
    print('\n')

    scr = '''
    PCore BetaBin {
        al = 1
        be = 1

        p1 ~ beta(al, be)
        p2 ~ beta(al, be)
    }
    '''

    model1 = BetaBinBN(bayes_net_from_script(scr), exo={'n2': 20})
    model1.print()

    print('Domain:')
    for do in model1.Domain:
        print('\t', do)

    p1 = model1.sample_prior()
    print("Parameters: ", p1)
    print("Likelihood: ", model1.evaluate(p1))



