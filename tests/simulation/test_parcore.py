import unittest
import sims_pars as dag


class ParameterCoreCloneTest(unittest.TestCase):
    def test_clone(self):
        script = '''
        PCore Regression {
            x = 1
            y = x + 1
            z = y + 1
        }
        '''

        bn = dag.bayes_net_from_script(script)

        na = dag.NodeSet('a')
        nb = na.new_child('b', as_fixed=['x'])
        nb.new_child('c', as_fixed=['z'])

        sc = dag.as_simulation_core(bn, na)

        pc_a = sc.generate('A')
        pc_b = pc_a.breed('B', 'b')
        pc_c = pc_b.breed('C', 'c')

        pc_aa = pc_a.clone(copy_sc=True, include_children=True)
        pc_cc = pc_aa.find_descendant('B@C')

        self.assertEqual(pc_c['z'], 3)
        self.assertEqual(pc_cc['z'], 3)

        pc_aa.impulse({'x': 5})
        self.assertEqual(pc_c['z'], 3)
        self.assertEqual(pc_cc['z'], 7)

        pc_a.impulse({'x': 7})
        self.assertEqual(pc_c['z'], 9)
        self.assertEqual(pc_cc['z'], 7)

    def test_nullset(self):
        script = '''
                PCore Regression {
                    p ~ beta(1, 1)
                    x ~ norm(0, 1)
                }
                '''
        bn = dag.bayes_net_from_script(script)
        sc = dag.as_simulation_core(bn)
        sc.deep_print()
        p = sc.generate()

        print(p.Locus)


if __name__ == '__main__':
    unittest.main()
