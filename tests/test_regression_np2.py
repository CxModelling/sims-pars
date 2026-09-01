"""Import every public submodule so NumPy 2 / pydantic 2 breakage surfaces
as a test failure rather than a runtime surprise."""
import importlib
import pkgutil

import pytest

import sims_pars

# Subpackages with heavy optional third-party deps or known-legacy internals
# that are intentionally not part of the supported import surface yet.
_SKIP_PREFIXES = (
    'sims_pars.data',          # legacy: still imports the predecessor `epidag`
    'sims_pars.fit.hme',       # optional extra: gpflow / tensorflow
)


def _modules():
    for m in pkgutil.walk_packages(sims_pars.__path__, 'sims_pars.'):
        if any(m.name == p or m.name.startswith(p + '.') for p in _SKIP_PREFIXES):
            continue
        yield m.name


@pytest.mark.parametrize('name', list(_modules()))
def test_submodule_imports(name):
    importlib.import_module(name)
