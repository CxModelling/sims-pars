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


def test_distribution_loci_parses_once_not_per_draw(monkeypatch):
    """A draw with changing parents must not re-parse the spec or regenerate a
    pydantic schema — the plan is built once in __init__."""
    import sims_pars.bayesnet.loci as loci_mod

    loci = DistributionLoci('v', 'norm(mu, sd)')  # the one legitimate parse

    def boom(*a, **k):
        raise AssertionError('re-parsed the distribution spec during a draw')

    monkeypatch.setattr(loci_mod, 'parse_function', boom)
    monkeypatch.setattr(loci_mod, 'complete_function', boom)

    for i in range(20):
        loci.get_distribution({'mu': float(i), 'sd': 1.0 + i})


def test_sp_double_does_not_freeze_the_scipy_distribution():
    import scipy.stats as sts

    from sims_pars.prob import parse_distribution

    d = parse_distribution('norm(mean=0, sd=1)')
    # holds the module-level family + kwds, not a frozen rv_frozen instance
    assert d.Family is sts.norm
    assert d.Kwds == {'loc': 0, 'scale': 1}


def test_datamodel_domain_is_cached():
    from sims_pars.fit.toys import get_betabin
    m = get_betabin((7, 14))
    assert m.Domain is m.Domain
