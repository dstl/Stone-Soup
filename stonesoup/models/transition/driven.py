from stonesoup.models.transition.base_driver import Latents
from .base import LinearDrivenTransitionModel
from ..base import TimeVariantModel
from ...base import Property
from datetime import timedelta
from scipy.linalg import expm
from scipy.integrate import quad_vec
from stonesoup.types.array import CovarianceMatrices, CovarianceMatrix, StateVector, StateVectors
from abc import abstractmethod
from typing import Callable
import numpy as np


class NumericalIntegratedModel:
    def _integrate(self, func: Callable[..., np.ndarray], a: np.ndarray, b: np.ndarray) -> np.ndarray:
        res, err = quad_vec(func, a=a, b=b)
        return res, err

    @abstractmethod
    def _eAdt(self, dt: float) -> np.ndarray :
        pass
    

class Process(LinearDrivenTransitionModel, TimeVariantModel):
    """
    Simple process model
    """

    def matrix(self, **kwargs) -> np.ndarray:
        return np.eye(1)

    def ext_input(self, **kwargs) -> np.ndarray:
        return np.zeros((1, 1))

    def ft(self, dt: float, jtimes: np.ndarray, **kwargs) -> np.ndarray:
        return np.ones_like(jtimes)[..., None, None]  # (n_jumps, n_samples, 1, 1)

    def e_ft(self, dt: float, **kwargs) -> np.ndarray:
        return dt * np.ones((1, 1))  # (1, 1)

    # def ft2(self, dt: float, jtimes: np.ndarray, **kwargs) -> np.ndarray:
    #     return np.ones_like(jtimes)[..., None, None] # (n_jumps, n_samples, 1, 1)

    # def e_ft2(self, dt: float, **kwargs) -> np.ndarray:
    #     return dt * np.eye(1)

    def e_gt(self, dt: float, **kwargs) -> np.ndarray:
        return dt * np.ones((1, 1))

    # def e_gt2(self, dt: float, **kwargs) -> np.ndarray:
    #     return dt * np.eye(1)


class Langevin(LinearDrivenTransitionModel, TimeVariantModel):

    theta: float = Property(doc="Theta parameter.")

    def matrix(self, time_interval: timedelta, **kwargs) -> np.ndarray:
        dt = time_interval.total_seconds()
        eA0 = np.array([[0, 1.0 / -self.theta], [0.0, 1.0]])
        eA1 = np.array([[1, 1.0 / self.theta], [0.0, 0.0]])
        eAdt = np.exp(-self.theta * dt) * eA0 + eA1
        # exp_A_delta_t
        return eAdt  # (m, m)

    def ft(self, dt: float, jtimes: np.ndarray, **kwargs) -> np.ndarray:
        """
        Summing terms here in the inner function would be more efficient,
        as returning a scalar faster than a vector. That being said, the
        summation is part of the point process simulation and the driver
        is responsible for it, not the model.
        """
        v1 = np.array([[1.0 / -self.theta], [1.0]])  # (m, 1)
        v2 = np.array([[1.0 / self.theta], [0.0]])
        term1 = np.exp(-self.theta * (dt - jtimes))[..., None, None]  # (n_jumps, n_samples, 1, 1)
        term2 = np.ones_like(jtimes)[..., None, None]
        return term1 * v1 + term2 * v2  # (n_jumps, n_samples, m, 1)

    # def ft2(self, dt: float, jtimes: np.ndarray, **kwargs) -> np.ndarray:
    #     M1 = np.array([[1.0 / (-self.theta**2), 1.0 / -self.theta], [1.0 / -self.theta, 1.0]])
    #     M2 = np.array([[-2.0 / (-self.theta**2), -1.0 / -self.theta], [-1.0 / -self.theta, 0.0]])
    #     M3 = np.array([[1.0 / (-self.theta**2), 0.0], [0.0, 0.0]])
    #     term1 = np.exp(2 * -self.theta * (dt - jtimes)).reshape(-1, 1, 1)
    #     term2 = np.exp(-self.theta * (dt - jtimes)).reshape(-1, 1, 1)
    #     term3 = np.ones_like(jtimes).reshape(-1, 1, 1)
    #     return term1 * M1 + term2 * M2 + term3 * M3

    def e_ft(self, dt: float, **kwargs) -> np.ndarray:
        v1 = np.array([[1.0 / -self.theta], [1.0]])
        v2 = np.array([[1.0 / self.theta], [0.0]])
        term1 = (np.exp(-self.theta * dt) - 1.0) / (-self.theta) * v1
        term2 = dt * v2
        return term1 + term2  # (m, 1)

    # def E_ft2(self, dt: float, **kwargs) -> np.ndarray:
    #     M1 = np.array([[1.0 / (-self.theta**2), 1.0 / -self.theta], [1.0 / -self.theta, 1.0]])
    #     M2 = np.array([[-2.0 / (-self.theta**2), -1.0 / -self.theta], [-1.0 / -self.theta, 0.0]])
    #     M3 = np.array([[1.0 / (-self.theta**2), 0.0], [0.0, 0.0]])
    #     term1 = (np.exp(2.0 * -self.theta * dt) - 1.0) / (2.0 * -self.theta) * M1
    #     term2 = (np.exp(-self.theta * dt) - 1.0) / -self.theta * M2
    #     term3 = dt * M3
    #     return term1 + term2 + term3

    def e_gt(self, dt: float, **kwargs) -> np.ndarray:
        v1 = np.array([[1.0 / -self.theta], [1.0]])
        v2 = np.array([[1.0 / self.theta], [0.0]])
        term1 = (np.exp(-self.theta * dt) - 1.0) / (-self.theta) * v1
        term2 = dt * v2
        return term1 + term2  # (m, 1)

    # def E_gt2(self, dt: float, **kwargs) -> np.ndarray:
    #     M1 = np.array([[1.0 / (self.theta**2), 1.0 / -self.theta], [1.0 / -self.theta, 1.0]])
    #     M2 = np.array([[-2.0 / (-self.theta**2), -1.0 / -self.theta], [-1.0 / -self.theta, 0.0]])
    #     M3 = np.array([[1.0 / (-self.theta**2), 0.0], [0.0, 0.0]])
    #     term1 = (np.exp(2.0 * -self.theta * dt) - 1.0) / (2.0 * -self.theta) * M1
    #     term2 = (np.exp(-self.theta * dt) - 1.0) / -self.theta * M2
    #     term3 = dt * M3
    #     return term1 + term2 + term3



class EquilibriumRevertingVelocity(LinearDrivenTransitionModel, TimeVariantModel, NumericalIntegratedModel):
    eta: float = Property(doc="eta parameter.")
    theta: float = Property(doc="eta parameter.")
    rho: float = Property(doc="rho parameter.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.control_matrix = self.eta * np.eye(self.ndim)
        # self.control_input = np.array([0.0, self.p]).reshape((2, 1))
        self.A = np.array([[0, 1.0], [-self.eta, -self.theta]])
        self.h = np.array([0.0, 1.0]).reshape((2, 1))
        self.g = np.array([0.0, 1.0]).reshape((2, 1))
        self.b = self.eta * np.array([0.0, self.rho]).reshape((2, 1))

    def matrix(self, time_interval: timedelta, **kwargs) -> np.ndarray:
        dt = time_interval.total_seconds()
        return expm(self.A * dt)

    def _eAdt(self, dt: float):
        return expm((self.A * dt))

    def rvs(self, time_interval: timedelta, num_samples: int = 1, latents: Latents | None = None, **kwargs) -> StateVector | StateVectors:
        dt = time_interval.total_seconds()
        func = lambda u: self._eAdt(dt - u) @ self.b
        bias, _ = self._integrate(func=func, a=0, b=dt) 
        return super().rvs(time_interval, num_samples, latents, **kwargs) + bias
    
    def mean(self, latents: Latents, time_interval: timedelta, **kwargs) -> StateVector | StateVectors:
        dt = time_interval.total_seconds()
        func = lambda u: self._eAdt(dt - u) @ self.g
        bias, _ = self._integrate(func=func, a=0, b=dt) 
        return super().mean(latents, time_interval, **kwargs) + bias
 
    def ft(self, dt: float, jtimes: np.ndarray, **kwargs) -> np.ndarray:
        return expm((self._eAdt(dt - jtimes[..., None, None]))) @ self.h


    def e_ft(self, dt: float, **kwargs) -> np.ndarray:
        func = lambda u: self._eAdt(dt - u) @ self.h
        integral, _ = self._integrate(func=func, a=0, b=dt)
        return integral 
    
    def e_gt(self, dt: float, **kwargs) -> np.ndarray:
        func = lambda u: self._eAdt(dt - u) @ self.g
        integral, _ = self._integrate(func=func, a=0, b=dt)
        return integral 
