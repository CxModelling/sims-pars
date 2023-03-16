from abc import ABCMeta, abstractmethod
import gpflow
import numpy as np

__all__ = ['AbsEmulator', 'GPREmulator']


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


class GPREmulator(AbsEmulator):
    def train(self, xs, ys):
        xs = np.array(xs)
        ys = np.array([[y[self.Output]] for y in ys], dtype=float)
        self.GP = gpflow.models.GPR(data=(xs, ys), kernel=self.Kernel)
        opt = gpflow.optimizers.Scipy()
        opt.minimize(self.GP.training_loss, self.GP.trainable_variables, options=self.Opt)

    def predict(self, xs) -> tuple[list, list]:
        assert self.GP is not None

        mean, var = self.GP.predict_f(xs)
        mean, var = mean.numpy(), var.numpy()
        return mean, var


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    X = np.array(
        [
            [0.865], [0.666], [0.804], [0.771], [0.147], [0.866], [0.007], [0.026],
            [0.171], [0.889], [0.243], [0.028],
        ]
    )
    Y = np.array(
        [
            [1.57], [3.48], [3.12], [3.91], [3.07], [1.35], [3.80], [3.82], [3.49],
            [1.30], [4.00], [3.82],
        ]
    )

    plt.plot(X, Y, "kx", mew=2)

    model = gpflow.models.GPR(
        (X, Y),
        kernel=gpflow.kernels.RBF(), mean_function=None
    )

    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables, options=dict())

    Xplot = np.linspace(-0.1, 1.1, 100)[:, None]

    f_mean, f_var = model.predict_f(Xplot, full_cov=False)
    y_mean, y_var = model.predict_y(Xplot)

    f_lower = f_mean - 1.96 * np.sqrt(f_var)
    f_upper = f_mean + 1.96 * np.sqrt(f_var)
    y_lower = y_mean - 1.96 * np.sqrt(y_var)
    y_upper = y_mean + 1.96 * np.sqrt(y_var)

    plt.plot(X, Y, "kx", mew=2, label="input data")
    plt.plot(Xplot, f_mean, "-", color="C0", label="mean")
    plt.plot(Xplot, f_lower, "--", color="C0", label="f 95% confidence")
    plt.plot(Xplot, f_upper, "--", color="C0")
    plt.fill_between(
        Xplot[:, 0], f_lower[:, 0], f_upper[:, 0], color="C0", alpha=0.1
    )
    plt.plot(Xplot, y_lower, ".", color="C0", label="Y 95% confidence")
    plt.plot(Xplot, y_upper, ".", color="C0")
    plt.fill_between(
        Xplot[:, 0], y_lower[:, 0], y_upper[:, 0], color="C0", alpha=0.1
    )
    plt.legend()
    plt.show()
