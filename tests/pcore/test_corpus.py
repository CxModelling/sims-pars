"""The compatibility oracle: the new front end must produce a network
structurally identical to the legacy regex parser for every PCore script that
exists in the repo and the notebooks.

Divergences are enumerated in ``EXPECTED_ERRORS`` — each is a place where the
legacy parser crashed or silently accepted garbage and the new one reports it.
"""
import pathlib

import pytest

from sims_pars.bayesnet import bayes_net_from_script
from sims_pars.pcore import DiagnosticError, compile_script

CORPUS = sorted((pathlib.Path(__file__).parent / "corpus").glob("*.pcore"))

# scripts the legacy parser cannot handle (it raises) — the new parser must
# report an error diagnostic instead of a network.
EXPECTED_ERRORS = {"06_Cyclic"}


def _canon(bn) -> dict:
    js = bn.to_json()
    return {
        "name": js["Name"],
        "order": set(js["Order"]),
        "exo": set(js["Exo"]),
        "roots": set(js["Roots"]),
        "leaves": set(js["Leaves"]),
        "nodes": {
            n["Name"]: (n["Type"], n.get("Def"), tuple(sorted(n.get("Parents", []))))
            for n in js["Nodes"]
        },
    }


def _is_topological(bn) -> bool:
    order = {n: i for i, n in enumerate(bn.Order)}
    g = bn.DAG
    return all(order[u] < order[v] for u, v in g.edges())


@pytest.mark.parametrize("path", CORPUS, ids=lambda p: p.stem)
def test_matches_legacy(path):
    src = path.read_text()

    if path.stem in EXPECTED_ERRORS:
        with pytest.raises((AttributeError, DiagnosticError)):
            bayes_net_from_script(src)
        with pytest.raises(DiagnosticError):
            compile_script(src, strict=True)
        return

    legacy = bayes_net_from_script(src)
    new = compile_script(src, strict=True)

    assert _canon(new) == _canon(legacy)
    assert set(new.Order) == set(legacy.Order)
    assert _is_topological(new)


def test_corpus_is_not_empty():
    assert len(CORPUS) >= 20
