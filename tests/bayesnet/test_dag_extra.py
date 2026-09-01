import unittest

from sims_pars.bayesnet.dag import (
    DAG,
    get_minimal_nodes,
    get_offsprings,
    get_sufficient_nodes,
)


def _diamond():
    g = DAG()
    g.add_edge('A', 'B')
    g.add_edge('A', 'C')
    g.add_edge('B', 'D')
    g.add_edge('C', 'D')
    return g


class DagStructureTest(unittest.TestCase):
    def test_children_and_descendants(self):
        g = _diamond()
        self.assertSetEqual(g.children('A'), {'B', 'C'})
        self.assertSetEqual(g.descendants('A'), {'B', 'C', 'D'})

    def test_roots_and_leaves(self):
        g = _diamond()
        self.assertListEqual(g.roots(), ['A'])
        self.assertListEqual(g.leaves(), ['D'])

    def test_order_is_topological(self):
        order = _diamond().order()
        self.assertEqual(order[0], 'A')
        self.assertEqual(order[-1], 'D')

    def test_acyclic_check(self):
        g = _diamond()
        self.assertTrue(g.check_acyclic())
        g.add_edge('D', 'A')
        self.assertFalse(g.check_acyclic())

    def test_up_and_downstream(self):
        g = _diamond()
        self.assertSetEqual(g.upstream(['D']), {'A', 'B', 'C'})
        self.assertSetEqual(g.downstream(['A']), {'B', 'C', 'D'})


class DagQueryTest(unittest.TestCase):
    def test_sufficient_nodes(self):
        self.assertSetEqual(get_sufficient_nodes(_diamond(), ['D']), {'A', 'B', 'C', 'D'})

    def test_minimal_nodes_drops_given(self):
        minimal = get_minimal_nodes(_diamond(), ['D'], given=['B', 'C'])
        self.assertNotIn('A', minimal)
        self.assertNotIn('B', minimal)

    def test_get_offsprings(self):
        self.assertSetEqual(get_offsprings(_diamond(), ['B']), {'D'})


if __name__ == '__main__':
    unittest.main()
