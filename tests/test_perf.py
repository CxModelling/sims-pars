"""Guards for the efficiency fixes made during the 3.x upgrade."""
import types

from sims_pars.bayesnet.loci import DistributionLoci
from sims_pars.util import parse_math_expression


def test_math_expression_precompiles_code_object():
    me = parse_math_expression('x + y / 2')
    assert isinstance(me._code, types.CodeType)
    # repeated evaluation must not rebuild the code object
    before = me._code
    me.execute({'x': 1, 'y': 2})
    assert me._code is before


def test_distribution_loci_reuses_frozen_dist_for_fixed_parents():
    loci = DistributionLoci('v', 'norm(mu, 1)')
    d1 = loci.get_distribution({'mu': 0.0})
    d2 = loci.get_distribution({'mu': 0.0})
    assert d1 is d2
    # a changed parent value invalidates the cache
    d3 = loci.get_distribution({'mu': 5.0})
    assert d3 is not d1


def test_distribution_loci_reuses_dist_with_no_parents():
    loci = DistributionLoci('v', 'norm(0, 1)')
    assert loci.get_distribution() is loci.get_distribution()


def test_datamodel_domain_is_cached():
    from sims_pars.fit.toys import get_betabin
    m = get_betabin((7, 14))
    assert m.Domain is m.Domain
