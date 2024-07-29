from datetime import timedelta
from typing import Union, Optional, Tuple, Callable, Dict, NamedTuple, Generator
from collections import namedtuple
from scipy.stats import multivariate_normal
from abc import abstractmethod
from ..base import Base, Property
from ..types.array import StateVector, StateVectors, CovarianceMatrix, CovarianceMatrices
import numpy as np


class Driver(Base):
    pass


class NoiseCase(Driver):
    pass


class TruncatedCase(NoiseCase):
    pass


class GaussianResidualApproxCase(NoiseCase):
    pass


class PartialGaussianResidualApproxCase(NoiseCase):
    pass


class Latents:
    def __init__(self, num_samples: int) -> None:
        self.store: Dict[Driver, NamedTuple] = dict()
        self.Data = namedtuple("Data", ["sizes", "times"])
        self._num_samples = num_samples

    def exists(self, driver) -> bool:
        return driver in self.store

    def add(self, driver: Driver, jsizes: np.ndarray, jtimes: np.ndarray) -> None:
        assert jsizes.shape == jtimes.shape
        assert jsizes.shape[1] == self._num_samples
        data = self.Data(jsizes, jtimes)
        self.store[driver] = data

    def sizes(self, driver) -> np.ndarray:
        assert driver in self.store
        # dimensions of sizes are (n_jumps, n_samples)
        return self.store[driver].sizes

    def times(self, driver) -> np.ndarray:
        assert driver in self.store
        # dimensions of times are (n_times, n_samples)
        return self.store[driver].times

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @num_samples.setter
    def num_samples(self, num_samples) -> None:
        self._num_samples = num_samples


class LevyDriver(Driver):
    """Driver type

    Base/Abstract class for all conditional Gaussian noise driving processes."""

    seed: Optional[int] = Property(default=None, doc="Seed for random number generation")
    c: np.double = Property(doc="Truncation parameter, expected no. jumps per unit time.")
    noise_case: NoiseCase = Property(
        default=GaussianResidualApproxCase(),
        doc="Cases for compensating residuals from series truncation",
    )
    mu_W: float = Property(default=0.0, doc="Default Gaussian mean")
    sigma_W2: float = Property(default=1.0, doc="Default Gaussian variance")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.random_state = np.random.default_rng(self.seed)

    def rvs(
        self,
        mean: StateVector,
        covar: CovarianceMatrix,
        random_state: Generator,
        num_samples: int = 1,
        **kwargs
    ) -> Union[StateVector, StateVectors]:
        """
        returns driving noise term
        """
        if random_state is None:
            random_state = self.random_state
        noise = random_state.multivariate_normal(
            mean.flatten(), covar, size=num_samples
        )
        noise = noise.T
        if num_samples == 1:
            return noise.view(StateVector)
        else:
            return noise.view(StateVectors)

    @abstractmethod
    def characteristic_func():
        pass

    @abstractmethod
    def _centering(self, e_ft: np.ndarray, truncation: float) -> StateVector:
        pass

    @abstractmethod
    def _hfunc(self, epochs: np.ndarray) -> np.ndarray:
        """H function"""
        pass

    @abstractmethod
    def _thinning_probabilities(self, jsizes: np.ndarray) -> np.ndarray:
        """Calculate thinning probabilities for accept-reject sampling"""
        pass

    @abstractmethod
    def _jump_power(self, jszies: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def _first_moment(self, truncation: float) -> float:
        pass

    @abstractmethod
    def _second_moment(self, truncation: float) -> float:
        pass

    @abstractmethod
    def _residual_covar(
        self, e_ft: np.ndarray, truncation: float, mu_W: float, sigma_W2: float
    ) -> CovarianceMatrix:
        pass

    def _residual_mean(self, e_ft: np.ndarray, truncation: float, mu_W: float) -> CovarianceMatrix:
        if isinstance(self.noise_case, TruncatedCase):
            m = e_ft.shape[0]
            r_mean = np.zeros((m, 1))
        elif isinstance(self.noise_case, GaussianResidualApproxCase) or isinstance(
            self.noise_case, PartialGaussianResidualApproxCase
        ):
            r_mean = e_ft * mu_W  # (m, 1)
        else:
            raise AttributeError("invalid noise case")
        return self._first_moment(truncation=truncation) * r_mean  # (m, 1)

    def _accept_reject(self, jsizes: np.ndarray, random_state: Generator) -> np.ndarray:
        probabilities = self._thinning_probabilities(jsizes)
        u = random_state.uniform(low=0.0, high=1.0, size=probabilities.shape)
        jsizes = np.where(u <= probabilities, jsizes, 0)
        return jsizes

    def sample_latents(self, dt: float, num_samples: int, random_state: Optional[Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
        if random_state is None:
            random_state = self.random_state
        # Sample latents pairs
        epochs = random_state.exponential(scale=1 / dt, size=(int(self.c * dt), num_samples))
        epochs = epochs.cumsum(axis=0)

        # Accept reject sampling
        jsizes = self._hfunc(epochs=epochs)
        jsizes = self._accept_reject(jsizes=jsizes, random_state=random_state)
        # Generate jump times
        jtimes = random_state.uniform(low=0.0, high=dt, size=jsizes.shape)
        return jsizes, jtimes

    def mean(
        self,
        latents: Latents,
        ft_func: Callable[..., np.ndarray],
        e_ft_func: Callable[..., np.ndarray],
        dt: float,
        mu_W: Optional[float]=None,
        **kwargs
    ) -> StateVector | StateVectors:
        """Computes a num_samples of mean vectors"""
        mu_W = np.atleast_2d(self.mu_W) if mu_W is None else np.atleast_2d(mu_W)

        jtimes = latents.times(driver=self)  # (n_jumps, n_samples)
        jsizes = latents.sizes(driver=self)  # (n_jumps, n_samples)
        num_samples = latents.num_samples
        assert(jsizes.shape[1] == (num_samples) and jtimes.shape[1] == (num_samples))
        truncation = self.c * dt
        ft = ft_func(dt=dt, jtimes=jtimes)  # (n_jumps, n_samples, m, 1)
        series = np.sum(jsizes[..., None, None] * ft, axis=0)  # (n_samples, m, 1)
        m = series * mu_W

        e_ft = e_ft_func(dt=dt)  # (m, 1)
        residual_mean = self._residual_mean(e_ft=e_ft, mu_W=mu_W, truncation=truncation)[None, ...]
        centering = dt * self._centering(e_ft=e_ft, mu_W=mu_W, truncation=truncation)[None, ...]
        mean = m - centering + residual_mean
        if num_samples == 1:
            return mean[0].view(StateVector)
        else:
            return mean.view(StateVectors)

    def covar(
        self,
        latents: Latents,
        ft_func: Callable[..., np.ndarray],
        e_ft_func: Callable[..., np.ndarray],
        dt: float,
        mu_W: Optional[float] = None,
        sigma_W2: Optional[float] = None,
        **kwargs
    ) -> CovarianceMatrix | CovarianceMatrices:
        """Computes covariance matrix / matrices"""
        mu_W = np.atleast_2d(self.mu_W) if mu_W is None else np.atleast_2d(mu_W)
        sigma_W2 = np.atleast_2d(self.sigma_W2) if sigma_W2 is None else np.atleast_2d(sigma_W2)     

        jsizes = self._jump_power(latents.sizes(driver=self))  # (n_jumps, n_samples)
        jtimes = latents.times(driver=self)
        num_samples = latents.num_samples
        assert(jsizes.shape[1] == (num_samples) and jtimes.shape[1] == (num_samples))

        truncation = self._hfunc(self.c * dt)

        ft = ft_func(dt=dt, jtimes=jtimes)  # (n_jumps, n_samples, m, 1)
        ft2 = np.einsum("ijkl, ijml -> ijkm", ft, ft)  # (n_jumps, n_samples, m, m)
        series = np.sum(jsizes[..., None, None] * ft2, axis=0)  # (n_samples, m, m)
        s = sigma_W2 * series

        e_ft = e_ft_func(dt=dt)  # (m, 1)
        residual_cov = self._residual_covar(e_ft=e_ft, mu_W=mu_W, sigma_W2=sigma_W2, truncation=truncation)
        covar = s + residual_cov
        if num_samples == 1:
            return covar[0].view(CovarianceMatrix)  # (m, m)
        else:
            return covar.view(CovarianceMatrices)  # (n_samples, m, m)


class NormalSigmaMeanDriver(LevyDriver):
    def _jump_power(self, jsizes: np.ndarray) -> np.ndarray:
        return jsizes**2

    def _residual_covar(
        self, e_ft: np.ndarray, truncation: float, mu_W: float, sigma_W2: float, **kwargs
    ) -> CovarianceMatrix:
        mu_W = mu_W
        sigma_W2 = sigma_W2
        if isinstance(self.noise_case, TruncatedCase):
            m = e_ft.shape[0]
            r_cov = np.zeros((m, m))
        elif isinstance(self.noise_case, GaussianResidualApproxCase):
            r_cov = (
                e_ft @ e_ft.T * self._second_moment(truncation=truncation) * (mu_W**2 + sigma_W2)
            )
        elif isinstance(self.noise_case, PartialGaussianResidualApproxCase):
            r_cov = e_ft @ e_ft.T * self._second_moment(truncation=truncation) * sigma_W2
        else:
            raise AttributeError("Invalid noise case.")
        return r_cov  # (m, m)


class NormalVarianceMeanDriver(LevyDriver):
    def _jump_power(self, jsizes: np.ndarray) -> np.ndarray:
        return jsizes

    def _centering(
        self, e_ft: np.ndarray, truncation: float, mu_W: float
    ) -> StateVector:
        m = e_ft.shape[0]
        return np.zeros((m, 1))

    def _residual_covar(
        self, e_ft: np.ndarray, truncation: float, mu_W: float, sigma_W2: float, **kwargs
    ) -> CovarianceMatrix:
        mu_W = mu_W
        sigma_W2 = sigma_W2
        if isinstance(self.noise_case, TruncatedCase):
            m = e_ft.shape[0]
            r_cov = np.zeros((m, m))
        elif isinstance(self.noise_case, GaussianResidualApproxCase):
            r_cov = (
                e_ft
                @ e_ft.T
                * (
                    self._second_moment(truncation=truncation) * mu_W**2
                    + self._first_moment(truncation=truncation) * sigma_W2
                )
            )
        elif isinstance(self.noise_case, PartialGaussianResidualApproxCase):
            r_cov = e_ft @ e_ft.T * self._first_moment(truncation=truncation) * sigma_W2
        else:
            raise AttributeError("Invalid noise case.")
        return r_cov  # (m, m)
