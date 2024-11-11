from typing import Callable, Generator, Optional, Union, override

import numpy as np
from numpy.typing import NDArray
from scipy.special import gamma as gammafnc
from scipy.special import gammainc

from stonesoup.base import Property
from stonesoup.models.base_driver import (ConditionallyGaussianDriver,
                                          LevyDriver, NoiseCase, NormalSigmaMeanDriver, NormalVarianceMeanDriver)
from stonesoup.types.array import (CovarianceMatrices, CovarianceMatrix,
                                   StateVector, StateVectors)


def incgammal(s: float, x: float) -> float:  # Helper function
    return gammainc(s, x) * gammafnc(s)


class GaussianDriver(LevyDriver):

    mu_W: float = Property(default=0.0, doc="Default Gaussian mean")
    sigma_W2: float = Property(default=1.0, doc="Default Gaussian variance")

    def sample_latents(self, dt: float, num_samples: int, random_state: Optional[Generator] = None) -> np.array:
        return np.zeros((1, num_samples)), np.zeros(1, num_samples)
    
    def characteristic_func(
        self, mu_W: Optional[float] = None, sigma_W2: Optional[float] = None, **kwargs
    ) -> Callable[[float], complex]:
        if mu_W is None:
            mu_W = self.mu_W
        if sigma_W2 is None:
            sigma_W2 = self.sigma_W2

        def inner(w: float):
            return np.exp(-1j * w * mu_W - 0.5 * sigma_W2 * w**2)

        return inner

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
            return mean.view(StateVector)
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
        sigma_W2 = (
            np.atleast_2d(self.sigma_W2) if sigma_W2 is None else np.atleast_2d(sigma_W2)
        )
        covar = sigma_W2 * e_ft @ e_ft.T
        if num_samples == 1:
            return covar.view(CovarianceMatrix)
        else:
            covar = np.tile(covar, (num_samples, 1, 1))
            return covar.view(CovarianceMatrices)


    def rvs(
        self,
        mean: StateVector,
        covar: CovarianceMatrix,
        random_state: Optional[Generator] = None,
        num_samples: int = 1,
        **kwargs
    ) -> Union[StateVector, StateVectors]:
        """Returns the noise sampled via the Levy driving process.

        Args:
            mean (StateVector): Mean of the conditionally Gaussian Levy density.
            covar (CovarianceMatrix): Covariance of the conditionally Gaussian
                                      Levy density.
            random_state (Generator, optional): Random state (seed) to use. Defaults
                                                to None.
            num_samples (int, optional): Number of noise samples to generate.
                                         Defaults to 1.

        Returns:
            Union[StateVector, StateVectors]: Single or multiple noise term as vectors.
        """
        if random_state is None:
            random_state = self.random_state
        noise = random_state.multivariate_normal(mean.flatten(), covar, size=num_samples)
        noise = noise.T
        if num_samples == 1:
            return noise.view(StateVector)
        else:
            return noise.view(StateVectors)


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

    def _residual_mean(
        self, e_ft: np.ndarray, truncation: float, mu_W: float
    ) -> StateVector:
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

    def characteristic_func(self):
        # TODO
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

    @override
    def _thinning_probabilities(self, jsizes: np.ndarray) -> np.ndarray:
        return (1.0 + self.beta * jsizes) * np.exp(
            -self.beta * jsizes
        )  # (n_jumps, n_samples)

    def _residual_covar(
        self, e_ft: np.ndarray, truncation: float, mu_W: float, sigma_W2: float
    ) -> CovarianceMatrix:
        m = e_ft.shape[0]
        return np.zeros((m, m))

    def _residual_mean(
        self,
        e_ft: np.ndarray,
        truncation: float,
        mu_W: float,
    ) -> CovarianceMatrix:
        m = e_ft.shape[0]
        return np.zeros((m, 1))

    def characteristic_func(self):
        # TODO
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

    @override
    def _thinning_probabilities(self, jsizes: np.ndarray) -> np.ndarray:
        return np.exp(-self.beta * jsizes)  # (n_jumps, n_samples)

    def characteristic_func(self):
        # TODO
        raise NotImplementedError
