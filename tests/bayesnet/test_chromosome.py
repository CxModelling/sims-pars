import unittest
from sims_pars.bayesnet import Chromosome, BayesianNetwork

__author__ = 'TimeWz667'


class ChromosomeTest(unittest.TestCase):
    def test_chr(self):
        cms = Chromosome({'A': 1}, -5)
        self.assertTrue(cms.is_evaluated())
        self.assertEqual(cms['A'], 1)

    def test_impact(self):
        bn = BayesianNetwork('Test')
        bn.append_from_definition('A=1')
        bn.append_from_definition('B=A+4')

        cms = Chromosome({'A': 1, 'B': 5}, -5)
        cms.impulse({'A': 5}, bn)
        self.assertEqual(cms['A'], 5)
        self.assertEqual(cms['B'], 9)

    def test_clone(self):
        cms = Chromosome({'A': 1}, -5)

        cms_copy = cms.clone()
        self.assertEqual(cms_copy.LogProb, -5)

    def test_impulse_clone(self):
        bn = BayesianNetwork('Test')
        bn.append_from_definition('A=1')
        bn.append_from_definition('B=A+4')

        cms = Chromosome({'A': 1, 'B': 5}, -5)
        cms_copy = cms.clone()

        cms.impulse({'A': 5}, bn)

        self.assertEqual(cms['A'], 5)
        self.assertEqual(cms['B'], 9)
        self.assertEqual(cms_copy['A'], 1)
        self.assertEqual(cms_copy['B'], 5)


if __name__ == '__main__':
    unittest.main()
