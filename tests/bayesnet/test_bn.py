import unittest
import sims_pars as dag


class BayesNetTest(unittest.TestCase):
    def setUp(self):
        self.BN1 = dag.bayes_net_from_script('''
        PCore A {
            A = 5
            B = A * 4
            C = B + B2
            D = pow(C)
        }
        ''')

        self.BN2 = dag.bayes_net_from_script('''
        PCore B2 {
            C = B
            D = pow(C)
        }
        ''')

    def test_rvroots(self):
        bn = dag.bayes_net_from_script('''
        PCore C {
            A = 2
            B ~ beta(3, A)
            C ~ binom(prob=B, size=5)
            D ~ unif(0, 1)
        }
        
        ''')
        self.assertCountEqual(bn.RVRoots, ['D', 'B'])

    def test_merge(self):
        bn3 = self.BN1.merge('B3', self.BN2)
        self.assertNotIn('B2', bn3)
        self.assertCountEqual(self.BN1.DAG.ancestors('D'), ['A', 'B', 'B2', 'C'])
        self.assertCountEqual(self.BN2.DAG.ancestors('D'), ['B', 'C'])
        self.assertCountEqual(bn3.DAG.ancestors('D'), ['A', 'B', 'C'])


if __name__ == '__main__':
    unittest.main()
