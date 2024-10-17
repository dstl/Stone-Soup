from datetime import timedelta
import numpy as np
from numpy import ndarray
from scipy.special import gammainc, kv, hankel1
from scipy.special import gamma as gammafnc
from stonesoup.types.array import CovarianceMatrix, StateVector, StateVectors, CovarianceMatrices
from .base_driver import LevyDriver, NormalSigmaMeanDriver, NormalVarianceMeanDriver
from ..base import Property
from typing import Optional, Callable, Tuple


def incgammal(s: float, x: float) -> float:  # Helper function
    return gammainc(s, x) * gammafnc(s)


class GaussianDriver(LevyDriver):
    @property
    def c(self):
        raise NotImplementedError("No associated truncation parameter with Gaussian driver.")

    @property
    def noise_case(self):
        raise NotImplementedError("No associated noise case with Gaussian driver.")

    def sample_latents(self, dt:float, num_samples: int, random_state: np.random.RandomState) -> Tuple[ndarray, ndarray]:
        return np.zeros((1,1)), np.zeros((1,1))

    def _centering(self, *args, **kwargs) -> StateVector:
        raise NotImplementedError

    def _first_moment(self, mu_W: float, *args, **kwargs) -> float:
        return mu_W

    def _second_moment(self, mu_W: float, sigma_W2: float, *args, **kwargs) -> float:
        return sigma_W2 + (mu_W**2)

    def _jump_power(self, *args, **kwargs) -> np.ndarray:
        return NotImplementedError

    def _hfunc(self, *args, **kwargs) -> np.ndarray:
        return NotImplementedError

    def _residual_covar(self, *args, **kwargs) -> CovarianceMatrix:
        return NotImplementedError

    def _residual_mean(self, *args, **kwargs) -> CovarianceMatrix:
        return NotImplementedError

    def _thinning_probabilities(self, *args, **kwargs) -> np.ndarray:
        return NotImplementedError

    def characteristic_func(
        self, mu_W: Optional[float] = None, sigma_W2: Optional[float] = None, *args, **kwargs
    ) -> np.ndarray:
        mu_W = np.atleast_2d(self.mu_W) if mu_W is None else np.atleast_2d(mu_W)
        sigma_W2 = np.atleast_2d(self.sigma_W2) if sigma_W2 is None else np.atleast_2d(sigma_W2)     

        return lambda w: np.exp(-0.5 * w**2 * sigma_W2 + 1j * w * mu_W)

    def mean(
        self,
        e_ft_func: Callable[..., np.ndarray],
        dt: float,
        mu_W: Optional[float] = None,
        num_samples: int = 1,
        **kwargs
    ) -> StateVector:
        mu_W = np.atleast_2d(self.mu_W) if mu_W is None else np.atleast_2d(mu_W)
        e_ft = e_ft_func(dt=dt)
        mean = mu_W * e_ft
        if num_samples == 1:
            return mean[0].view(StateVector)
        else:
            mean = np.tile(mean, (num_samples, 1, 1))
            return mean.view(StateVectors)


    def covar(
        self,
        e_ft_func: Callable[..., np.ndarray],
        dt: float,
        sigma_W2: Optional[float] = None,
        num_samples: int = 1,
        **kwargs
    ) -> CovarianceMatrix:
        e_ft = e_ft_func(dt=dt)
        sigma_W2 = np.atleast_2d(self.sigma_W2) if sigma_W2 is None else np.atleast_2d(sigma_W2)     

        covar = sigma_W2 * np.einsum("ijk, ilk -> ijl", e_ft, e_ft)
        if num_samples == 1:
            return covar[0].view(CovarianceMatrix)
        else:
            covar = np.tile(covar, (num_samples, 1, 1))
            return covar.view(CovarianceMatrices)


class AlphaStableNSMDriver(NormalSigmaMeanDriver):
    alpha: float = Property(doc="Alpha parameter.")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if (
            np.isclose(self.alpha, 0.0)
            or np.isclose(self.alpha, 1.0)
            or np.isclose(self.alpha, 2.0)
        ):
            raise AttributeError("alpha must be 0 < alpha < 1 or 1 < alpha < 2.")

    def _hfunc(self, epochs: np.ndarray) -> np.ndarray:
        return np.power(epochs, -1.0 / self.alpha)

    def _first_moment(self, **kwargs) -> float:
        return self.alpha / (1.0 - self.alpha) * np.power(self.c, 1.0 - 1.0 / self.alpha)

    def _second_moment(self, **kwargs) -> float:
        return self.alpha / (2.0 - self.alpha) * np.power(self.c, 1.0 - 2.0 / self.alpha)

    def _residual_mean(self, e_ft: np.ndarray, truncation: float, mu_W: float) -> StateVector:
        if 1 < self.alpha < 2:
            m = e_ft.shape[0]
            r_mean = np.zeros((m, 1))
            return r_mean
        return super()._residual_mean(e_ft=e_ft, mu_W=mu_W, truncation=truncation)

    def _centering(self, e_ft: np.ndarray, truncation: float, mu_W: float) -> StateVector:
        if 1 < self.alpha < 2:
            term = e_ft * mu_W  # (m, 1)
            return -self._first_moment(truncation=truncation) * term  # (m, 1)
        elif 0 < self.alpha < 1:
            m = e_ft.shape[0]
            return np.zeros((m, 1))
        else:
            raise AttributeError("alpha must be 0 < alpha < 2")

    def _thinning_probabilities(self, jsizes: np.ndarray) -> np.ndarray:
        # accept all
        return np.ones_like(jsizes)  # (n_jumps, n_samples)

    def characteristic_func(self):
        # TODO: Use inverse FFT method
        raise NotImplementedError


class GammaNVMDriver(NormalVarianceMeanDriver):
    nu: float = Property(doc="Scale parameter")
    beta: float = Property(doc="Shape parameter")

    def _hfunc(self, epochs: np.ndarray) -> np.ndarray:
        return 1.0 / (self.beta * (np.exp(epochs / self.nu) - 1.0))

    def _first_moment(self, truncation: float) -> float:
        return (self.nu / self.beta) * incgammal(1.0, self.beta * truncation)

    def _second_moment(self, truncation: float) -> float:
        return (self.nu / self.beta**2) * incgammal(2.0, self.beta * truncation)

    def _thinning_probabilities(self, jsizes: np.ndarray) -> np.ndarray:
        return (1.0 + self.beta * jsizes) * np.exp(-self.beta * jsizes)  # (n_jumps, n_samples)

    def _residual_covar(
        self, e_ft: np.ndarray, truncation: float,  mu_W: float, sigma_W2: float
    ) -> CovarianceMatrix:
        m = e_ft.shape[0]
        return np.zeros((m, m))

    def _residual_mean(
        self, e_ft: np.ndarray, truncation: float,  mu_W: float,
    ) -> CovarianceMatrix:
        m = e_ft.shape[0]
        return np.zeros((m, 1))

    def characteristic_func(self):
        # TODO: Use inverse FFT method
        raise NotImplementedError


class TemperedStableNVMDriver(NormalVarianceMeanDriver):
    alpha: float = Property(doc="Alpha parameter")
    beta: float = Property(doc="Shape parameter")

    def _hfunc(self, epochs: np.ndarray) -> np.ndarray:
        return np.power(epochs, -1.0 / self.alpha)

    def _first_moment(self, truncation: float) -> float:
        return (self.alpha * self.beta ** (self.alpha - 1.0)) * incgammal(
            1.0 - self.alpha, self.beta * truncation
        )

    def _second_moment(self, truncation: float) -> float:
        return (self.alpha * self.beta ** (self.alpha - 2.0)) * incgammal(
            2.0 - self.alpha, self.beta * truncation
        )

    def _thinning_probabilities(self, jsizes: np.ndarray) -> np.ndarray:
        return np.exp(-self.beta * jsizes)  # (n_jumps, n_samples)

    def characteristic_func(self):
        # TODO: Use inverse FFT method
        raise NotImplementedError
