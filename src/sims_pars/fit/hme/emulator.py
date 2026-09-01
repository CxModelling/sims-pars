"""Gaussian-process emulators for Bayesian history matching.

Backed by GPyTorch (CPU only — no device/GPU options). Each emulator learns one
scalar output of the simulator and returns a predictive mean and variance for
the latent function, matching the ``predict_f`` semantics of the previous
gpflow implementation.
"""
from __future__ import annotations

import warnings
from abc import ABCMeta, abstractmethod

import gpytorch
import numpy as np
import torch

__all__ = ['AbsEmulator', 'GPREmulator']

_DTYPE = torch.float64   # GP regression needs double precision for a stable Cholesky


def _tensor(a) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a, dtype=float), dtype=_DTYPE)


class AbsEmulator(metaclass=ABCMeta):
    def __init__(self, output, kernel=None, maxiter: int = 100, lr: float = 0.1, **kwargs):
        self.Output = output
        self.Kernel = kernel
        self.Opt = {'maxiter': maxiter, 'lr': lr, **kwargs}
        self.GP = None

    @abstractmethod
    def train(self, xs, ys) -> None:
        ...

    @abstractmethod
    def predict(self, xs) -> tuple[np.ndarray, np.ndarray]:
        ...


class _ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        base = kernel if kernel is not None else gpytorch.kernels.RBFKernel()
        self.covar_module = gpytorch.kernels.ScaleKernel(base)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class GPREmulator(AbsEmulator):
    """Exact GP regression emulator for a single simulator output."""

    def train(self, xs, ys):
        x = _tensor(xs)
        y = _tensor([row[self.Output] for row in ys])

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(_DTYPE)
        model = _ExactGP(x, y, likelihood, kernel=self.Kernel).to(_DTYPE)
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.Opt['lr'])
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        for _ in range(int(self.Opt['maxiter'])):
            optimizer.zero_grad()
            loss = -mll(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        likelihood.eval()
        self.GP = model

    def predict(self, xs) -> tuple[np.ndarray, np.ndarray]:
        """Predictive mean and variance of the latent function, as 1-D arrays."""
        assert self.GP is not None, "emulator must be trained before predict()"
        x = _tensor(xs)
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), warnings.catch_warnings():
            warnings.simplefilter("ignore", gpytorch.utils.warnings.GPInputWarning)
            pred = self.GP(x)
            mean = pred.mean.cpu().numpy().ravel()
            var = pred.variance.cpu().numpy().ravel()
        return mean, var


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    X = rng.random((20, 1))
    Y = [{'y': float(3.0 + np.sin(6 * x[0]))} for x in X]

    emu = GPREmulator('y', maxiter=80)
    emu.train(X, Y)

    Xp = np.linspace(-0.1, 1.1, 10)[:, None]
    m, v = emu.predict(Xp)
    for xp, mi, vi in zip(Xp[:, 0], m, v):
        print(f"x={xp:+.2f}  mean={mi:+.3f}  sd={np.sqrt(vi):.3f}")
