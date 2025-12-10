# sde/euler_maruyama.py
import jax
import jax.lax as lax
import jax.numpy as np
from jax import random
from typing import Union
from coefficients import Coefficient

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
        X0 = np.asarray(self._X_0, dtype=np.float32)
        delta = np.asarray(self.delta, dtype=np.float32)

        noise_shape = (self._n_steps, dim) + self._state_shape
        dW = random.normal(key, shape=noise_shape, dtype=np.float32) * np.sqrt(delta)

        Y0 = np.broadcast_to(X0, (dim,) + self._state_shape)

        drift_batched = lambda y, t: self._drift.get_value(y[None, ...], t)[0]
        diff_batched = lambda y, t: self._diffusion.get_value(y[None, ...], t)[0]

        drift_batched = jax.vmap(drift_batched, in_axes=(0, None))
        diff_batched = jax.vmap(diff_batched, in_axes=(0, None))

        t_seq = np.asarray(self.t[:-1], dtype=np.float32)

        def body(Y_n, x):

            t_n, dW_n = x
            mu = drift_batched(Y_n, t_n)
            sigma = diff_batched(Y_n, t_n)
            Y_next = Y_n + mu * delta + sigma * dW_n
            return Y_next, Y_next
        
        xs = (t_seq, dW)

        _, Y_steps = lax.scan(body, Y0, xs)

        Y_steps = np.transpose(Y_steps, (1, 0) + tuple(range(2, Y_steps.ndim)))
        
        Y0_expanded = np.expand_dims(Y0, axis = 1)
        Y_full = np.concatenate([Y0_expanded, Y_steps], axis=1)

        return Y_full

    def compute_numerical_approximation(self, key: jax.Array) -> np.ndarray:
        solve = jax.jit(self._solve_numerical_approximation, static_argnames=("dim",))
        self.Y = solve(dim=self._n_sim, key=key)
        return self.Y