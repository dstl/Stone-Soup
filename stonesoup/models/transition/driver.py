import numpy as np
from scipy.special import gammainc, gamma
from stonesoup.types.array import CovarianceMatrix, StateVector
from .base_driver import NormalSigmaMeanDriver, NormalVarianceMeanDriver
from ...base import Property


def incgammal(s: float, x: float) -> float: # Helper function
    return gammainc(s, x) * gamma(s)

class AlphaStableNSMDriver(NormalSigmaMeanDriver):
    alpha: float = Property(doc="Alpha parameter.")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if np.isclose(self.alpha, 0.0) or np.isclose(self.alpha, 1.0) or np.isclose(self.alpha, 2.0):
            raise AttributeError("alpha must be 0 < alpha < 1 or 1 < alpha < 2.")

    def _hfunc(self, epochs: np.ndarray) -> np.ndarray:
        return np.power(epochs, -1.0 / self.alpha)

    def _first_moment(self) -> np.ndarray:
        return self.alpha / (1. - self.alpha) * np.power(self.c, 1. - 1. / self.alpha)
    
    def _second_moment(self) -> np.ndarray:
        return self.alpha / (2. - self.alpha) * np.power(self.c, 1. - 2. / self.alpha)

    def _residual_mean(self, e_ft: np.ndarray) -> CovarianceMatrix:
        if 1 < self.alpha < 2:
            m = e_ft.shape[0]
            r_mean = np.zeros((m, 1))
            return r_mean
        return super()._residual_mean(e_ft)
    
    def _centering(self, e_ft: np.ndarray) -> StateVector:
        if 1 < self.alpha < 2:
            term = e_ft * self.mu_W # (m, 1)
            return self._first_moment() * term # (m, 1)
        elif 0 < self.alpha < 1:
            m = e_ft.shape[0]
            return np.zeros((m, 1))
        else:
            raise AttributeError("alpha must be 0 < alpha < 2")

    def _thinning_probabilities(self, jsizes: np.ndarray) -> np.ndarray:
        # accept all
        return np.ones_like(jsizes) # (n_jumps, n_samples)
    

class GammaNVMDriver(NormalVarianceMeanDriver):
    nu: float = Property(doc="Scale parameter")
    beta: float = Property(doc="Shape parameter")

    def _hfunc(self, epochs: np.ndarray) -> np.ndarray:
        return 1. / (self.beta * (np.exp(epochs / self.nu) - 1.))

    def _first_moment(self) -> np.ndarray:
        truncation = self._hfunc(self.c)
        return (self.nu / self.beta) * incgammal(1., self.beta * truncation)
    
    def _second_moment(self) -> np.ndarray:
        truncation = self._hfunc(self.c)
        return (self.nu / self.beta ** 2) * incgammal(2., self.beta * truncation)

    def _thinning_probabilities(self, jsizes: np.ndarray) -> np.ndarray:
        return (1. + self.beta * jsizes) * np.exp(-self.beta * jsizes) # (n_jumps, n_samples)