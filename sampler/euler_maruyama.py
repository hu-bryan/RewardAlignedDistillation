# sde/euler_maruyama.py
import jax
import jax.numpy as np
from jax import random
from typing import Union
from .coefficients import Coefficient

class EulerMaruyama:
    def __init__(
        self,
        t_0: float,
        t_n: float,
        n_steps: int,
        X_0: Union[float, np.ndarray],
        drift: Coefficient,
        diffusion: Coefficient,
        n_sim: int,
    ):
        self._t_0 = t_0
        self._t_n = t_n
        self._n_steps = n_steps

        self._X_0 = np.asarray(X_0)
        self._state_shape = self._X_0.shape

        self._drift = drift
        self._diffusion = diffusion

        self._n_sim = n_sim

        self.Y = None
        self._compute_discretisation()

    @property
    def n_sim(self):
        return self._n_sim

    @n_sim.setter
    def n_sim(self, value: int):
        if value > 0:
            self._n_sim = value
        else:
            raise ValueError("Number of simulations must be positive.")

    @property
    def n_steps(self):
        return self._n_steps

    @n_steps.setter
    def n_steps(self, value: int):
        if value > 0:
            self._n_steps = value
            self._compute_discretisation()
        else:
            raise ValueError("Number of steps must be positive.")

    def _compute_discretisation(self) -> None:
        self.t = np.linspace(self._t_0, self._t_n, self._n_steps + 1)
        self.delta = (self._t_n - self._t_0) / self._n_steps

    def _allocate_Y(self, dim: int) -> np.ndarray:
        Y_shape = (dim, self._n_steps + 1) + self._state_shape
        Y = np.zeros(Y_shape, dtype=np.float32)
        init = np.broadcast_to(self._X_0, (dim,) + self._state_shape)
        Y = Y.at[:, 0].set(init)
        return Y

    def _solve_numerical_approximation(self, dim: int, key: jax.Array) -> np.ndarray:
        Y = self._allocate_Y(dim=dim)
        noise_shape = (dim, self._n_steps) + self._state_shape
        dW = random.normal(key, shape=noise_shape) * np.sqrt(self.delta)

        for n in range(self._n_steps):
            tau_n = self.t[n]
            Y_n = Y[:, n]
            mu = self._drift.get_value(X=Y_n, t=tau_n)
            sigma = self._diffusion.get_value(X=Y_n, t=tau_n)
            incr = mu * self.delta + sigma * dW[:, n]
            Y = Y.at[:, n + 1].set(Y_n + incr)

        return Y

    def compute_numerical_approximation(self, key: jax.Array) -> np.ndarray:
        self.Y = self._solve_numerical_approximation(dim=self._n_sim, key=key)
        return self.Y
