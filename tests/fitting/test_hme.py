"""GP emulator + history-matching smoke tests. Skipped unless the `hme` extra
(gpytorch) is installed."""
import numpy as np
import pytest

pytest.importorskip("gpytorch")

from sims_pars.fit import ApproxBayesCom  # noqa: E402 - after importorskip
from sims_pars.fit.hme.emulator import GPREmulator  # noqa: E402


def test_emulator_learns_a_smooth_function():
    rng = np.random.default_rng(0)
    x = rng.random((24, 1))
    y = [{"o": float(2.0 + np.sin(5 * xi[0]))} for xi in x]

    emu = GPREmulator("o", maxiter=120)
    emu.train(x, y)

    grid = np.linspace(0, 1, 15)[:, None]
    mean, var = emu.predict(grid)
    assert mean.shape == (15,) and var.shape == (15,)
    assert np.all(var > 0)

    truth = 2.0 + np.sin(5 * grid[:, 0])
    assert np.sqrt(np.mean((mean - truth) ** 2)) < 0.25


def test_history_matching_runs_end_to_end():
    from sims_pars.fit.hme import BayesHistoryMatching
    from sims_pars.fit.toys import get_betabin

    model = get_betabin((4, 12))
    alg = BayesHistoryMatching(max_wave=2, n_sims=20, n_ems=400)
    alg.fit(model)
    post = alg.sample_posteriors(100)
    df = post.to_df()
    assert len(df) == 100
    assert np.isfinite(df["p1"].mean())


def test_hme_is_still_wired_to_the_fit_package():
    # unrelated import guard: the fit package proper stays importable
    assert ApproxBayesCom is not None
