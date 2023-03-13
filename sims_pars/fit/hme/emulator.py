from abc import ABCMeta, abstractmethod
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np

__all__ = ['AbsEmulator', 'GPREmulator']


class AbsEmulator(metaclass=ABCMeta):
    def __init__(self, output, kernel=None, **kwargs):
        self.Output = output
        if kernel is None:
            self.Kernel = RBF(length_scale=1e-10)
        else:
            self.Kernel = kernel
        self.Opt = dict(kwargs)
        self.GP = None

    @abstractmethod
    def train(self, xs, ys) -> None:
        pass

    @abstractmethod
    def predict(self, xs) -> tuple[np.ndarray, np.ndarray]:
        pass


class GPREmulator(AbsEmulator):
    def train(self, xs, ys):
        ys = np.array([[y[self.Output]] for y in ys])
        self.GP = GaussianProcessRegressor(kernel=self.Kernel, n_restarts_optimizer=5, **self.Opt)
        self.GP.fit(xs, ys)

    def predict(self, xs) -> tuple[list, list]:
        assert self.GP is not None
        mean, var = self.GP.predict(xs, return_std=True)
        return mean, var
