"""Phase 3 (type annotations) and Phase 4 (plates, include composition)."""
import pytest

from sims_pars.pcore import check, compile_script, parse


# --- Phase 3: type annotations -------------------------------------------

def test_annotation_on_a_matching_constant_is_silent():
    src = '''
PCore Ann {
    a: float = 3
    b: int = 3
    n: bool = True
    v: vector = [1, 2, 3]
    s: simplex = [0.5, 0.5]
}
'''
    diags = check(src)
    assert diags == []
    bn = compile_script(src)
    assert set(bn.Order) == {'a', 'b', 'n', 'v', 's'}


def test_annotation_mismatch_is_a_warning_not_an_error():
    src = '''
PCore Ann {
    c: int = 3.5
}
'''
    diags = check(src)
    assert len(diags) == 1
    assert diags[0].code == "E0301"
    assert not diags[0].is_error
    # a warning still builds the node
    bn = compile_script(src)
    assert 'c' in bn.Order


def test_unknown_type_name_is_an_error_with_a_hint():
    src = '''
PCore Ann {
    a: floot = 3
}
'''
    diags = check(src)
    assert any(d.code == "E0300" and d.is_error for d in diags)
    assert any('float' in (d.hint or '') for d in diags)


def test_annotation_on_a_distribution_or_function_is_recorded_but_not_checked():
    # neither a '~' node nor a '=' expression with free names has a value at
    # parse time, so no E0301 fires even though 'int' looks implausible for p.
    src = '''
PCore Ann {
    n = 4
    p: int ~ beta(1, 1)
    q: int = p * n
}
'''
    diags = check(src)
    assert diags == []


# --- Phase 4: plates -------------------------------------------------------

def test_plate_expands_one_node_per_iteration():
    src = '''
PCore Plated {
    mu = 2
    for i in 1..3 {
        x ~ norm(mu, 1)
        y = x[i] * 2
    }
    total = x_1 + x_2 + x_3
}
'''
    program = parse(src)
    assert program.diagnostics == []
    bn = program.to_network()
    assert set(bn.Order) == {'mu', 'x_1', 'x_2', 'x_3', 'y_1', 'y_2', 'y_3', 'total'}
    assert repr(bn['y_2']) == 'y_2 = x_2*2'


def test_plate_bare_loop_variable_becomes_the_iteration_value():
    src = '''
PCore Plated {
    for i in 1..3 {
        c = i * 10
    }
}
'''
    bn = compile_script(src)
    from sims_pars.fn import sample
    vs = sample(bn)
    assert vs['c_1'] == 10 and vs['c_2'] == 20 and vs['c_3'] == 30


def test_nested_plates_compose():
    src = '''
PCore Nested {
    for j in 1..2 {
        for i in 1..2 {
            x = i + j
        }
    }
}
'''
    bn = compile_script(src)
    from sims_pars.fn import sample
    vs = sample(bn)
    assert vs['x_1_1'] == 2 and vs['x_1_2'] == 3
    assert vs['x_2_1'] == 3 and vs['x_2_2'] == 4


def test_empty_plate_range_is_a_warning_and_no_nodes():
    src = '''
PCore Empty {
    for i in 3..1 {
        x = i
    }
    a = 1
}
'''
    diags = check(src)
    assert len(diags) == 1 and diags[0].code == "E0128" and not diags[0].is_error
    bn = compile_script(src)
    assert bn.Order == ['a']


def test_include_inside_a_plate_is_rejected():
    src = '''
PCore Bad {
    for i in 1..2 {
        include "x.pcore"
    }
}
'''
    diags = check(src)
    assert any(d.code == "E0127" for d in diags)


# --- Phase 4: composition (include) ----------------------------------------

@pytest.fixture
def include_files(tmp_path):
    (tmp_path / 'priors.pcore').write_text('''
PCore Priors {
    al = 1
    be = 1
    p ~ beta(al, be)
}
''')
    (tmp_path / 'main.pcore').write_text('''
PCore Main {
    include "priors.pcore"
    x ~ binom(10, p)
}
''')
    return tmp_path


def test_include_splices_the_other_file_nodes_in(include_files):
    path = str(include_files / 'main.pcore')
    src = open(path).read()
    diags = check(src, path=path)
    assert diags == []
    bn = compile_script(src, path=path)
    assert set(bn.Order) == {'al', 'be', 'p', 'x'}


def test_include_missing_file_is_a_located_error(include_files):
    (include_files / 'broken.pcore').write_text('''
PCore Broken {
    include "nope.pcore"
    x = 1
}
''')
    path = str(include_files / 'broken.pcore')
    diags = check(open(path).read(), path=path)
    assert any(d.code == "E0250" for d in diags)


def test_include_cycle_is_detected(include_files):
    (include_files / 'a.pcore').write_text('PCore A {\n    include "b.pcore"\n    x = 1\n}\n')
    (include_files / 'b.pcore').write_text('PCore B {\n    include "a.pcore"\n    y = 2\n}\n')
    path = str(include_files / 'a.pcore')
    diags = check(open(path).read(), path=path)
    assert any(d.code == "E0254" for d in diags)


def test_include_without_a_path_resolves_relative_to_cwd(include_files, monkeypatch):
    monkeypatch.chdir(include_files)
    src = (include_files / 'main.pcore').read_text()
    bn = compile_script(src)  # no `path` -- falls back to os.getcwd()
    assert set(bn.Order) == {'al', 'be', 'p', 'x'}
