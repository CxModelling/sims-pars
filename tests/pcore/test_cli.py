from sims_pars.pcore.cli import main


def test_check_ok(tmp_path, capsys):
    f = tmp_path / "good.pcore"
    f.write_text("PCore M {\n a = 1\n b ~ norm(a, 1)\n}\n")
    assert main(["check", str(f)]) == 0
    assert "ok" in capsys.readouterr().out


def test_check_reports_and_exits_nonzero(tmp_path, capsys):
    f = tmp_path / "bad.pcore"
    f.write_text("PCore M {\n b ~ nrom(a, 1)\n}\n")
    assert main(["check", str(f)]) == 1
    out = capsys.readouterr().out
    assert "E0211" in out and "did you mean 'norm'" in out


def test_missing_file(tmp_path, capsys):
    assert main(["check", str(tmp_path / "nope.pcore")]) == 1
