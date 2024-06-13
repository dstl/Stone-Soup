from .linear import ConstantNthDerivative, NthDerivativeDecay, ConstantAcceleration, ConstantVelocity
from ...base import Property
from .base import ConditionalGaussianDriver
from functools import lru_cache
from scipy.integrate import quad
import numpy as np


class LevyNthDerivativeDecay(ConditionalGaussianDriver, NthDerivativeDecay):
    @classmethod
    @lru_cache()
    def _covardiscrete(cls, N, q, K, dt):
        covar = np.zeros((N + 1, N + 1))
        for k in range(0, N + 1):
            for l in range(0, N + 1):  # noqa: E741
                covar[k, l] = quad(cls._continouscovar, 0,
                                   dt, args=(N, K, k, l))[0]
        return covar * q 


class LevyConstantNthDerivative(ConditionalGaussianDriver, LevyNthDerivativeDecay):
    # Instead of inheriting ConstantNthDerivative,
    # Use NthDerivativeDecay with damping coeff = 0 
    # as temporary workaround because I required the
    # discrete covar method for returning the integrand
    
    # TODO: Inherit from ConstantNthDerivative and 
    # TODO: Write _covardiscrete(...) 

    """
    Non-Gaussian Constant Nth derivative model.
    """

    constant_derivative: int = Property(
        doc="The order of the derivative with respect to time to be kept constant, eg if 2 "
            "identical to constant acceleration")
    
    damping_coeff: float = Property(readonly=True, doc="The Nth derivative damping coefficient :math:`K`")

    @property
    def damping_coeff(self):
        return 0.0
    
    @property
    def decay_derivative(self):
        return self.constant_derivative


class LevyConstantVelocity(LevyConstantNthDerivative, ConstantVelocity):
    pass


class LevyConstantAcceleration(LevyConstantNthDerivative, ConstantVelocity):
    pass

