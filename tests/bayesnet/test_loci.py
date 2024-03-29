import unittest
import sims_pars.bayesnet.loci as lc


class LociTest(unittest.TestCase):
    def test_single_value(self):
        loci = lc.ValueLoci('v1', '0.5')
        self.assertDictEqual(loci.to_json(), {'Name': 'v1', 'Type': 'Value', 'Def': 0.5})
        self.assertEqual(loci.render(), 0.5)
        self.assertEqual(loci.evaluate(), 0)

    def test_exo_value(self):
        loci = lc.ExoValueLoci('v2')
        self.assertDictEqual(loci.to_json(), {'Name': 'v2', 'Type': 'ExoValue'})
        with self.assertRaises(KeyError):
            loci.render()
        self.assertEqual(loci.render({'v2': 0}), 0)
        self.assertEqual(loci.evaluate({'v2': 0}), 0)

    def test_random_variable(self):
        loci = lc.DistributionLoci('v3', 'norm(mu, 1)')
        self.assertEqual(loci.to_json()['Def'], 'norm(mean=mu, sd=1)')
        self.assertIsNotNone(loci.render({'mu': 0}))

    def test_function_variable(self):
        loci = lc.FunctionLoci('v4', 'x*(y + 5)')
        self.assertEqual(loci.to_json()['Def'], 'x*(y + 5)')
        self.assertEqual(loci({'x': 5, 'y': 3}), 40)

    def test_pseudo_variable(self):
        loci = lc.PseudoLoci('v5', 'x*(y + 5)')
        self.assertEqual(loci.to_json()['Def'], 'f(' + ', '.join(loci.Parents) + ')')
        with self.assertRaises(AttributeError):
            loci({'x': 5, 'y': 3})
        with self.assertRaises(AttributeError):
            loci.evaluate({'x': 5, 'y': 3})


if __name__ == '__main__':
    unittest.main()
