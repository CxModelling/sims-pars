import pandas as pd
from sims_pars.bayesnet.chromosome import Chromosome


__all__ = ['ParameterSet']


class ParameterSet:
    def __init__(self, alg='Prior'):
        self.Algorithm = alg
        self.Settings = dict()
        self.Notes = dict()
        self.ParameterList = list()
        self.DF = pd.DataFrame()

    def keep(self, k, note):
        self.Notes[k] = note

    def append(self, p):
        self.ParameterList.append(p)

    def finish(self):
        self.DF = Chromosome.to_data_frame(self.ParameterList)

    def to_json(self):
        return {
            'Algorithm': self.Algorithm,
            'Settings': dict(self.Settings),
            'Notes': dict(self.Notes),
            'Posterior': [pars.to_data() for pars in self.ParameterList]
        }

    def save_to_csv(self, file):
        return self.DF.to_csv(file)
