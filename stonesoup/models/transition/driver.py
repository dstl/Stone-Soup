import numpy as np

from stonesoup.types.array import CovarianceMatrix, StateVector
from .base_driver import NormalSigmaMeanDriver, NormalVarianceMeanDriver
from ...base import Property

class AlphaStableNSMDriver(NormalSigmaMeanDriver):
    alpha: float = Property(doc="Alpha parameter.")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if np.isclose(self.alpha, 0.0) or np.isclose(self.alpha, 1.0) or np.isclose(self.alpha, 2.0):
            raise AttributeError("alpha must be 0 < alpha < 1 or 1 < alpha < 2.")

    def _hfunc(self, epochs: np.ndarray) -> np.ndarray:
        return np.power(epochs, -1.0 / self.alpha)

    def _residual_mean(self, e_ft: np.ndarray) -> CovarianceMatrix:
        if 1 < self.alpha < 2 or self.noise_case == 1:
            r_mean = np.zeros((self.ndim_state, self.ndim_state))
        elif self.noise_case == 2 or self.noise_case == 3:
            r_mean = e_ft @ self.mu_W.T # (m, m)
        else:
            raise AttributeError("invalid noise case")
        return self.alpha / (1. - self.alpha) * np.power(self.c, 1. - 1. / self.alpha) * r_mean # (m, m)

    def _residual_cov(self, e_ft: np.ndarray) -> CovarianceMatrix:
        if self.noise_case == 1:
            r_sigma = np.zeros((self.ndim_state, self.ndim_state))
        elif self.noise_case == 2:
            tmp1 = e_ft @ self.mu_W.T 
            tmp2 = self.sigma_W @ e_ft
            r_sigma =  tmp1 @ tmp1.T + tmp2.T @ tmp2
        else:
            raise AttributeError("invalid noise case")
        return self.alpha / (2. - self.alpha) * np.power(self.c, 1. - 2. / self.alpha) * r_sigma # (m, m)
    
    def _centering(self, e_ft: np.ndarray) -> StateVector:
        if 1 < self.alpha < 2:
            term = e_ft @ self.mu_W.T # (m, m)
            return self.alpha / (1. - self.alpha) * np.power(self.c, 1. - 1. / self.alpha) * term # (m, m)
        elif 0 < self.alpha < 1:
            return np.zeros((self.ndim_state, self.ndim_state))
        else:
            raise AttributeError("alpha must be 0 < alpha < 2")

    def _thinning_probabilities(self, jsizes: np.ndarray) -> np.ndarray:
        # accept all
        return np.ones_like(jsizes) # (n_jumps, n_samples)