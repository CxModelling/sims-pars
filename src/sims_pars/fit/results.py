import pandas as pd

__all__ = ['ParameterSet']


class ParameterSet:
    def __init__(self, alg='Prior'):
        self.Algorithm = alg
        self.Settings = dict()
        self.Notes = dict()
        self.Particles = list()

    def __getitem__(self, item):
        return self.Notes[item]

    def __len__(self):
        return len(self.Particles)

    def keep(self, k, note):
        self.Notes[k] = note

    def append(self, p):
        self.Particles.append(p)

    def to_df(self, predictive=False):
        return pd.DataFrame([dict(pt.Pars) for pt in self.Particles])

    def to_json(self):
        return {
            'Algorithm': self.Algorithm,
            'Settings': dict(self.Settings),
            'Notes': dict(self.Notes),
            'Posterior': [dict(pt.Pars) for pt in self.Particles]
        }

    def to_pred_df(self):
        return pd.DataFrame([dict(pt['Ys']) for pt in self.Particles])
