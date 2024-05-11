from .base import LinearDrivenTransitionModel
from ..base import TimeVariantModel
from ...base import Property
from datetime import timedelta
import numpy as np


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
        eA0 = np.array([[0, 1.0 / self.theta], [0.0, 1.0]])
        eA1 = np.array([[1, -1.0 / self.theta], [0.0, 0.0]])
        eAdt = np.exp(self.theta * dt) * eA0 + eA1
        # exp_A_delta_t
        return eAdt  # (m, m)

    def ft(self, dt: float, jtimes: np.ndarray, **kwargs) -> np.ndarray:
        """
        Summing terms here in the inner function would be more efficient,
        as returning a scalar faster than a vector. That being said, the
        summation is part of the point process simulation and the driver
        is responsible for it, not the model.
        """
        v1 = np.array([[1.0 / self.theta], [1.0]])  # (m, 1)
        v2 = np.array([[-1.0 / self.theta], [0.0]])
        term1 = np.exp(self.theta * (dt - jtimes))[..., None, None]  # (n_jumps, n_samples, 1, 1)
        term2 = np.ones_like(jtimes)[..., None, None]
        return term1 * v1 + term2 * v2  # (n_jumps, n_samples, m, 1)

    # def ft2(self, dt: float, jtimes: np.ndarray, **kwargs) -> np.ndarray:
    #     M1 = np.array([[1.0 / (self.theta**2), 1.0 / self.theta], [1.0 / self.theta, 1.0]])
    #     M2 = np.array([[-2.0 / (self.theta**2), -1.0 / self.theta], [-1.0 / self.theta, 0.0]])
    #     M3 = np.array([[1.0 / (self.theta**2), 0.0], [0.0, 0.0]])
    #     term1 = np.exp(2 * self.theta * (dt - jtimes)).reshape(-1, 1, 1)
    #     term2 = np.exp(self.theta * (dt - jtimes)).reshape(-1, 1, 1)
    #     term3 = np.ones_like(jtimes).reshape(-1, 1, 1)
    #     return term1 * M1 + term2 * M2 + term3 * M3

    def e_ft(self, dt: float, **kwargs) -> np.ndarray:
        v1 = np.array([[1.0 / self.theta], [1.0]])
        v2 = np.array([[-1.0 / self.theta], [0.0]])
        term1 = (np.exp(self.theta * dt) - 1.0) / self.theta * v1
        term2 = dt * v2
        return term1 + term2  # (m, 1)

    # def E_ft2(self, dt: float, **kwargs) -> np.ndarray:
    #     M1 = np.array([[1.0 / (self.theta**2), 1.0 / self.theta], [1.0 / self.theta, 1.0]])
    #     M2 = np.array([[-2.0 / (self.theta**2), -1.0 / self.theta], [-1.0 / self.theta, 0.0]])
    #     M3 = np.array([[1.0 / (self.theta**2), 0.0], [0.0, 0.0]])
    #     term1 = (np.exp(2.0 * self.theta * dt) - 1.0) / (2.0 * self.theta) * M1
    #     term2 = (np.exp(self.theta * dt) - 1.0) / self.theta * M2
    #     term3 = dt * M3
    #     return term1 + term2 + term3

    def e_gt(self, dt: float, **kwargs) -> np.ndarray:
        v1 = np.array([[1.0 / self.theta], [1.0]])
        v2 = np.array([[-1.0 / self.theta], [0.0]])
        term1 = (np.exp(self.theta * dt) - 1.0) / self.theta * v1
        term2 = dt * v2
        return term1 + term2  # (m, 1)

    # def E_gt2(self, dt: float, **kwargs) -> np.ndarray:
    #     M1 = np.array([[1.0 / (self.theta**2), 1.0 / self.theta], [1.0 / self.theta, 1.0]])
    #     M2 = np.array([[-2.0 / (self.theta**2), -1.0 / self.theta], [-1.0 / self.theta, 0.0]])
    #     M3 = np.array([[1.0 / (self.theta**2), 0.0], [0.0, 0.0]])
    #     term1 = (np.exp(2.0 * self.theta * dt) - 1.0) / (2.0 * self.theta) * M1
    #     term2 = (np.exp(self.theta * dt) - 1.0) / self.theta * M2
    #     term3 = dt * M3
    #     return term1 + term2 + term3
