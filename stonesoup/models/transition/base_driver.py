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
    mu_W: np.ndarray = Property(doc="Gaussian mean vector")
    sigma_W2: np.ndarray = Property(doc="Gaussian diagonal covariance matrix")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._rng = np.random.default_rng(seed=self.seed)
        self.mu_W = StateVector(self.mu_W)
        self.sigma_W2 = CovarianceMatrix(self.sigma_W2)
        if self.sigma_W2.shape[0] != self.sigma_W2.shape[1]:
            raise AttributeError("covariance matrix sigma_W2 must be square")
        if self.mu_W.shape[0] != self.sigma_W2.shape[0]:
            raise AttributeError("ndim of mu_W must match sigma_W2")
        if np.any(np.linalg.eigvals(self.sigma_W2) <= 0):
            raise AttributeError("covariance not positive definite")
        self.sigma_W = np.linalg.cholesky(self.sigma_W2)

    @property
    def ndim_state(self) -> int:
        return self.mu_W.shape[0]
    
    def mean(self, e_gt_func: Callable[..., np.ndarray], dt:float, **kwargs) -> StateVector:
        e_gt = e_gt_func(dt=dt)
        return self.mu_W @ e_gt

    # def covar(self, e_gt2_func: Callable[..., np.ndarray], dt:float, **kwargs) -> CovarianceMatrix:
    #     e_gt2 = e_gt2_func(dt=dt)
    #     return self.sigma_W2 @ e_gt2

    def covar(self, e_gt_func: Callable[..., np.ndarray], dt:float, **kwargs) -> CovarianceMatrix:
        e_gt = e_gt_func(dt=dt)
        return e_gt @ self.sigma_W2 @ e_gt.T
    
    def rvs(
        self,
        mean: StateVector,
        covar: CovarianceMatrix,
        num_samples: int = 1,
        **kwargs
    ) -> Union[StateVector, StateVectors]:
        """
        returns driving noise term
        """
        print(mean.shape, covar.shape)
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
        assert(jsizes.shape == jtimes.shape)
        assert(jsizes.shape[1] == self._num_samples)
        data = self.Data(jsizes, jtimes)
        self.store[driver] = data

    def sizes(self, driver):
        assert driver in self.store
        # dimensions of sizes are (num_samples, n_jumps)
        return self.store[driver].sizes

    def times(self, driver):
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
        default=2, doc="Noise case, either 1, 2 or 3, refer to paper for more details."
    )

    @abstractmethod
    def _centering(self, e_ft: np.ndarray) -> StateVector:
        pass

    @abstractmethod
    def _residual_mean(self, e_ft: np.ndarray) -> CovarianceMatrix:
        pass

    @abstractmethod
    def _residual_cov(self, e_ft2: np.ndarray) -> CovarianceMatrix:
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
    def _jump_sizes(self, jszies: np.ndarray) -> np.ndarray:
        pass

    def _accept_reject(self, jsizes: np.ndarray) -> np.ndarray:
        probabilities = self._thinning_probabilities(jsizes)
        u = self._rng.uniform(low=0.0, high=1.0, size=probabilities.shape)
        jsizes = np.where(u < probabilities, jsizes, 0)
        return jsizes

    def sample_latents(self, dt: float, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample latents pairs"""
        epochs = self._rng.exponential(scale=1 / dt, size=(int(self.c * dt), num_samples))
        epochs = epochs.cumsum(axis=0)
        
        # Accept reject sampling
        jsizes = self._hfunc(epochs=epochs)
        jsizes = self._accept_reject(jsizes=jsizes)

        # Generate jump times
        jtimes = self._rng.uniform(low=0.0, high=dt, size=jsizes.shape)
        return jsizes, jtimes

    def mean(self, latents: Latents, ft_func: Callable[..., np.ndarray], e_ft_func: Callable[..., np.ndarray], dt:float,  **kwargs) -> StateVector:
        """Computes a num_samples of mean vectors"""
        jtimes = latents.times(driver=self) # (n_jumps, n_samples)
        jsizes = latents.sizes(driver=self) # (n_jumps, n_samples)
        num_samples = latents.num_samples
        
        ft = ft_func(dt=dt, jtimes=jtimes) # (n_jumps, n_samples, m, 1)
        e_ft = e_ft_func(dt=dt) # (m, 1)
        series = np.sum(jsizes[..., None, None] * ft, axis=0) # (n_samples, m, 1)
        m = series @ self.mu_W.T
        
        residual_mean = self._residual_mean(e_ft=e_ft)[None, ...]
        centering = self._centering(e_ft=e_ft)[None, ...]

        mean = m + centering + residual_mean
        if num_samples == 1:
            print(mean[0].shape)
            return mean[0].view(StateVector)
        else:
            return mean.view(StateVectors)

    def covar(self, latents: Latents, ft_func: Callable[..., np.ndarray], e_ft_func: Callable[..., np.ndarray], dt: float, **kwargs) -> CovarianceMatrix | CovarianceMatrices:
        """Computes covariance matrix / matrices"""
        jsizes = self._jump_sizes(latents.sizes(driver=self))
        jtimes = latents.times(driver=self)
        num_samples = latents.num_samples

        ft = ft_func(dt=dt, jtimes=jtimes)  # (n_jumps, n_samples, m, m)
        e_ft = e_ft_func(dt=dt) # (m, 1)
        series = np.sum(jsizes[..., None, None] * ft, axis=0) # (n_samples, m, 1)
        s = np.einsum("ijk, kl -> ijl", series, self.sigma_W2)  # (n_samples, m, m)
        s = np.einsum("ijk, ilk -> ijl", s, series)

        # residual_cov = self._residual_cov(e_ft2=e_ft2)
        residual_cov = self._residual_cov(e_ft=e_ft)
        covar = s + residual_cov
        if num_samples == 1:
            return covar[0].view(CovarianceMatrix) # (m, m)
        else:
            return covar.view(CovarianceMatrices) # (n_samples, m, m)

        


class NormalSigmaMeanDriver(ConditionalGaussianDriver):
    def _jump_sizes(self, jsizes: np.ndarray) -> np.ndarray:
        return jsizes
        
class NormalVarianceMeanDriver(ConditionalGaussianDriver):
    def _jump_sizes(self, jsizes: np.ndarray) -> np.ndarray:
        return np.sqrt(jsizes)
