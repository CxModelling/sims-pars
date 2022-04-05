import unittest
import sims_pars.util as dag
import sims_pars.prob as dist


class TestParseFunction(unittest.TestCase):
    def test_not_valid_function(self):
        with self.assertRaises(SyntaxError):
            dag.parse_function('exp[rate=0.4]')

    def test_one_argument(self):
        fn = dag.parse_function('exp(0.4)')
        self.assertEqual(fn.Function, 'exp')
        self.assertEqual(fn.get_arguments()[0]['value'], 0.4)

    def test_many_arguments(self):
        fn = dag.parse_function('gamma(0.4, 1/1000)')
        args = [arg['value'] for arg in fn.get_arguments()]
        self.assertListEqual(args, [0.4, 0.001])

    def test_key_argument(self):
        fn = dag.parse_function('gamma(0.4, scale=1/1000)')
        self.assertEqual(fn.get_arguments()[1]['key'], 'scale')

    def test_dict_argument(self):
        fn = dag.parse_function('cat(kv={"a": 4, "b": 3})')
        self.assertDictEqual({'a': 4, 'b': 3}, fn.get_arguments()[0]['value'])

    def test_list_argument(self):
        fn = dag.parse_function('list(vs=[2, "hh", 4])')
        self.assertListEqual(fn.get_arguments()[0]['value'], [2, "hh", 4])

    def test_two_step(self):
        fn = dag.parse_function('gamma(shape=0.01, rate=0.02)')
        self.assertEqual(fn.Function, 'gamma')
        arg_keys = [arg['key'] for arg in fn.get_arguments()]
        self.assertListEqual(arg_keys, ['shape', 'rate'])
        di = dist.parse_distribution(fn)
        self.assertEqual(di.mean(), 0.5)


if __name__ == '__main__':
    unittest.main()
