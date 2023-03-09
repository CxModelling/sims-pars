from abc import ABCMeta, abstractmethod
import numpy as np


class AbsEmulator(metaclass=ABCMeta):
    def __init__(self, output, kernel=None, **kwargs):
        self.Output = output
        if kernel is None:
            self.Kernel = gpflow.kernels.RBF()
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
