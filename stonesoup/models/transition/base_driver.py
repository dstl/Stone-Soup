from datetime import timedelta
from typing import Union, Optional, Tuple, Callable, Dict, NamedTuple
from collections import namedtuple
from abc import abstractmethod
from ...base import Base, Property
from ...types.array import StateVector, StateVectors, CovarianceMatrix, CovarianceMatrices
import numpy as np


class Driver(Base):
    pass


class GaussianDriver(Driver):
    """Driver type

    Base/Abstract class for all Gaussian noise driving processes."""

    seed: Optional[int] = Property(default=None, doc="Seed for random number generation")
    mu_W: float = Property(doc="Default Gaussian mean")
    sigma_W2: float = Property(doc="Default Gaussian variance")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._rng = np.random.default_rng(seed=self.seed)
        self.mu_W = np.atleast_2d(self.mu_W)
        self.sigma_W2 = np.atleast_2d(self.sigma_W2)
        self._mu_W_dict: dict[int, float] = {}
        self._sigma_W2_dict: dict[int, float] = {}
        self.set_params(None, self.mu_W, self.sigma_W2)
        assert self.mu_W.size == 1  # is float
        assert self.sigma_W2.size == 1  # is float

    def set_params(
        self, model_id: int, mu_W: Optional[float], sigma_W2: Optional[float]
    ) -> Tuple[float, float]:
        self._mu_W_dict[model_id] = self.mu_W if mu_W is None else np.atleast_2d(mu_W)
        self._sigma_W2_dict[model_id] = (
            self.sigma_W2 if sigma_W2 is None else np.atleast_2d(sigma_W2)
        )
        return self._mu_W_dict[model_id], self._sigma_W2_dict[model_id]

    def _mu_W(self, model_id: Optional[int] = None):
        return self._mu_W_dict[model_id]

    def _sigma_W2(self, model_id: Optional[int] = None):
        return self._sigma_W2_dict[model_id]

    def mean(
        self,
        e_gt_func: Callable[..., np.ndarray],
        dt: float,
        model_id: Optional[int] = None,
        **kwargs
    ) -> StateVector:
        e_gt = e_gt_func(dt=dt)
        return self._mu_W(model_id) * e_gt

    def covar(
        self,
        e_gt_func: Callable[..., np.ndarray],
        dt: float,
        model_id: Optional[int] = None,
        **kwargs
    ) -> CovarianceMatrix:
        e_gt = e_gt_func(dt=dt)
        return self._sigma_W2(model_id) * e_gt @ e_gt.T

    def rvs(
        self, mean: StateVector, covar: CovarianceMatrix, num_samples: int = 1, **kwargs
    ) -> Union[StateVector, StateVectors]:
        """
        returns driving noise term
        """
        # if np.any(np.linalg.eigvals(covar) < 0):
        #     # covar += np.diag(self._rng.normal(size=mean.ndim)) * 1e-6
        #     # print(covar)
        #     pass
        noise = self._rng.multivariate_normal(mean.flatten(), covar, size=num_samples)
        noise = noise.T
        if num_samples == 1:
            return noise.view(StateVector)
        else:
            return noise.view(StateVectors)


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
        # dimensions of sizes are (num_samples, n_jumps)
        return self.store[driver].sizes

    def times(self, driver) -> np.ndarray:
        assert driver in self.store
        # dimensions of times are (num_samples, n_jumps)
        return self.store[driver].times

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @num_samples.setter
    def num_samples(self, num_samples) -> None:
        self._num_samples = num_samples


class ConditionalGaussianDriver(GaussianDriver):
    c: np.double = Property(doc="Truncation parameter, expected no. jumps per unit time.")
    noise_case: int = Property(
        default=2,
        doc="Noise case must be either 1 (Truncated series), 2 (Gaussian residual approximation) or 3 (Partial Gaussian residual approximation), refer to paper for more details.",
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.noise_case not in [1, 2, 3]:
            raise AttributeError(
                "Noise case must be either: (1) Truncated series, (2) Gaussian residual approximation, (3) Partial Gaussian residual approximation"
            )

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
        self, e_ft: np.ndarray, truncation: float, model_id: Optional[int] = None
    ) -> CovarianceMatrix:
        pass

    def _residual_mean(
        self, e_ft: np.ndarray, truncation: float, model_id: Optional[int] = None
    ) -> CovarianceMatrix:
        if self.noise_case == 1:
            m = e_ft.shape[0]
            r_mean = np.zeros((m, 1))
        elif self.noise_case == 2 or self.noise_case == 3:
            r_mean = e_ft * self._mu_W(model_id)  # (m, 1)
        else:
            raise AttributeError("invalid noise case")
        # print(self._first_moment(truncation=truncation), r_mean)
        return self._first_moment(truncation=truncation) * r_mean  # (m, 1)

    def _accept_reject(self, jsizes: np.ndarray) -> np.ndarray:
        probabilities = self._thinning_probabilities(jsizes)
        u = self._rng.uniform(low=0.0, high=1.0, size=probabilities.shape)
        jsizes = np.where(u <= probabilities, jsizes, 0)
        # print(np.sum(np.where(jsizes == 0, 1, 0)))
        return jsizes

    def sample_latents(self, dt: float, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        # Sample latents pairs
        epochs = self._rng.exponential(scale=1 / dt, size=(int(self.c * dt), num_samples))
        epochs = epochs.cumsum(axis=0)

        # Accept reject sampling
        jsizes = self._hfunc(epochs=epochs)
        jsizes = self._accept_reject(jsizes=jsizes)
        # print(jsizes)
        # Generate jump times
        # print(jsizes)
        jtimes = self._rng.uniform(low=0.0, high=dt, size=jsizes.shape)
        return jsizes, jtimes

    def mean(
        self,
        latents: Latents,
        ft_func: Callable[..., np.ndarray],
        e_ft_func: Callable[..., np.ndarray],
        dt: float,
        model_id: Optional[int] = None,
        **kwargs
    ) -> StateVector | StateVectors:
        """Computes a num_samples of mean vectors"""
        jtimes = latents.times(driver=self)  # (n_jumps, n_samples)
        jsizes = latents.sizes(driver=self)  # (n_jumps, n_samples)
        num_samples = latents.num_samples
        truncation = self.c * dt
        ft = ft_func(dt=dt, jtimes=jtimes)  # (n_jumps, n_samples, m, 1)
        e_ft = e_ft_func(dt=dt)  # (m, 1)
        series = np.sum(jsizes[..., None, None] * ft, axis=0)  # (n_samples, m, 1)
        m = series * self._mu_W(model_id)

        residual_mean = self._residual_mean(e_ft=e_ft, model_id=model_id, truncation=truncation)[
            None, ...
        ]
        # residual_mean = 0
        centering = dt * self._centering(e_ft=e_ft, model_id=model_id, truncation=truncation)[None, ...]
        # print(residual_mean, centering)
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
        model_id: Optional[int] = None,
        **kwargs
    ) -> CovarianceMatrix | CovarianceMatrices:
        """Computes covariance matrix / matrices"""
        jsizes = self._jump_power(latents.sizes(driver=self))
        jtimes = latents.times(driver=self)
        num_samples = latents.num_samples
        truncation = self._hfunc(self.c * dt)

        ft = ft_func(dt=dt, jtimes=jtimes)  # (n_jumps, n_samples, m, 1)
        ft2 = np.einsum("ijkl, ijml -> ijkm", ft, ft)  # (n_samples, m, m)
        series = np.sum(jsizes[..., None, None] * ft2, axis=0)  # (n_samples, m, m)
        # series = np.einsum("ikl, iml -> ikm", series, series) # (n_samples, m, m)
        s = self._sigma_W2(model_id) * series

        e_ft = e_ft_func(dt=dt)  # (m, 1)
        residual_cov = self._residual_covar(e_ft=e_ft, model_id=model_id, truncation=truncation)
        covar = s + residual_cov
        if num_samples == 1:
            return covar[0].view(CovarianceMatrix)  # (m, m)
        else:
            return covar.view(CovarianceMatrices)  # (n_samples, m, m)


class NormalSigmaMeanDriver(ConditionalGaussianDriver):
    def _jump_power(self, jsizes: np.ndarray) -> np.ndarray:
        return jsizes**2

    def _residual_covar(
        self, e_ft: np.ndarray, truncation: float, model_id: Optional[int] = None
    ) -> CovarianceMatrix:
        mu_W = self._mu_W(model_id)
        sigma_W2 = self._sigma_W2(model_id)
        if self.noise_case == 1:
            m = e_ft.shape[0]
            r_cov = np.zeros((m, m))
        elif self.noise_case == 2:
            r_cov = (
                e_ft @ e_ft.T * self._second_moment(truncation=truncation) * (mu_W**2 + sigma_W2)
            )
        elif self.noise_case == 3:
            r_cov = e_ft @ e_ft.T * self._second_moment(truncation=truncation) * sigma_W2
        else:
            raise AttributeError("invalid noise case")
        return r_cov  # (m, m)


class NormalVarianceMeanDriver(ConditionalGaussianDriver):
    def _jump_power(self, jsizes: np.ndarray) -> np.ndarray:
        # return np.sqrt(jsizes)
        return jsizes

    def _centering(
        self, e_ft: np.ndarray, truncation: float, model_id: Optional[int] = None
    ) -> StateVector:
        m = e_ft.shape[0]
        return np.zeros((m, 1))

    def _residual_covar(
        self, e_ft: np.ndarray, truncation: float, model_id: Optional[int] = None
    ) -> CovarianceMatrix:
        mu_W = self._mu_W(model_id)
        sigma_W2 = self._sigma_W2(model_id)
        if self.noise_case == 1:
            m = e_ft.shape[0]
            r_cov = np.zeros((m, m))
        elif self.noise_case == 2:
            r_cov = (
                e_ft
                @ e_ft.T
                * (
                    self._second_moment(truncation=truncation) * mu_W**2
                    + self._first_moment(truncation=truncation) * sigma_W2
                )
            )
        elif self.noise_case == 3:
            r_cov = e_ft @ e_ft.T * self._first_moment(truncation=truncation) * sigma_W2
        else:
            raise AttributeError("invalid noise case")
        return r_cov  # (m, m)
