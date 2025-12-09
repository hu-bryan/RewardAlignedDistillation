# sde/coefficients.py
import jax.numpy as np
from abc import ABC, abstractmethod

class Coefficient(ABC):
    @abstractmethod
    def get_value(self, X: np.ndarray, t: float) -> np.ndarray:
        raise NotImplementedError

class LinearDrift(Coefficient):
    def __init__(self, a: float):
        self.a = a

    def get_value(self, X, t):
        return np.ones_like(X) * self.a * t

class MeanReversionDrift(Coefficient):
    def __init__(self, theta: float, mean: float):
        self.theta = theta
        self.mean = mean

    def get_value(self, X, t):
        return self.theta * (self.mean - X)

class ConstantDiffusion(Coefficient):
    def __init__(self, b: float):
        self.b = b

    def get_value(self, X, t):
        return np.ones_like(X) * self.b

class MultiplicativeNoiseDiffusion(Coefficient):
    def __init__(self, b: float):
        self.b = b

    def get_value(self, X, t):
        return self.b * X

class ConstantImageDiffusion(Coefficient):
    def __init__(self, b: float):
        self.b = b

    def get_value(self, X, t):
        return np.ones_like(X) * self.b
