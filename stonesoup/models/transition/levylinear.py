from matplotlib.pylab import RandomState
from stonesoup.models.base_driver import Latents
from stonesoup.types.array import CovarianceMatrices, CovarianceMatrix, StateVector, StateVectors
from ...base import Property
from ..base import LevyModel, LinearModel
from .base import CombinedLevyTransitionModel, LinearModel, TransitionModel, TimeVariantModel
from functools import lru_cache
import numpy as np
from datetime import timedelta
from scipy.integrate import quad_vec
from scipy.linalg import expm, block_diag
from typing import Optional, Union


class LinearLevyTransitionModel(TransitionModel, LinearModel, LevyModel):

    @property
    def ndim_state(self):
        """ndim_state getter method

        Returns
        -------
        : :class:`int`
            The number of model state dimensions.
        """

        return self.matrix().shape[0]


class CombinedLinearLevyTransitionModel(CombinedLevyTransitionModel, LinearModel):
    r"""Combine multiple models into a single model by stacking them.

    The assumption is that all models are Linear and Levy.
    Time Variant, and Time Invariant models can be combined together.
    If any of the models are time variant the keyword argument "time_interval"
    must be supplied to all methods
    """

    def matrix(self, **kwargs):
        """Model matrix :math:`F`

        Returns
        -------
        : :class:`numpy.ndarray` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
        """

        transition_matrices = [model.matrix(**kwargs) for model in self.model_list]
        return block_diag(*transition_matrices)


class LevyConstantNthDerivative(LinearLevyTransitionModel, TimeVariantModel):
    r"""Identical model to :class:`~.ConstantNthDerivative`, but driving white noise is now replaced with
    a Levy driving process.
    """

    constant_derivative: int = Property(
        doc="The order of the derivative with respect to time to be kept constant, eg if 2 "
        "identical to constant acceleration"
    )

    noise_diff_coeff: Optional[float] = Property(default=1.0, doc="The acceleration noise diffusion coefficient :math:`q`")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.A = self._init_A()
        self.h = self._init_h()

    def _init_A(self):
        N = self.constant_derivative
        mat = np.zeros((N + 1, N + 1))
        for i in range(N):
            mat[i, i + 1] = 1
        return mat

    def _init_h(self):
        N = self.constant_derivative
        mat = np.zeros((N + 1, 1))
        mat[N, 0] = 1
        return mat

    @property
    def ndim_state(self):
        return self.constant_derivative + 1
    
    def _integrand(self, dt: float, jtimes: np.ndarray) -> np.ndarray:
        delta = dt - jtimes[..., np.newaxis, np.newaxis]
        return expm(self.A * delta) @ self.h

    def matrix(self, time_interval, **kwargs):
        dt = time_interval.total_seconds()
        return expm(self.A * dt)

    def rvs(
        self,
        latents: Optional[Latents] = None,
        num_samples: int = 1,
        random_state: RandomState = None,
        **kwargs
    ) -> Union[StateVector, StateVectors]:
        coeff = self.noise_diff_coeff
        return super().rvs(latents, num_samples, random_state, **kwargs) * coeff


class LevyRandomWalk(LevyConstantNthDerivative):
    r"""This is a class implementation of a discrete, time-variant 1D
    Linear-Levy Random Walk Transition Model.

        The target is assumed to be (almost) stationary, where
        target velocity is modelled as white noise.
    """

    @property
    def constant_derivative(self):
        """For random walk, this is 0."""
        return 0


class LevyConstantVelocity(LevyConstantNthDerivative):
    r"""This is a class implementation of a discrete, time-variant 1D
    Linear-Levy Constant Velocity Transition Model.

    The target is assumed to move with (nearly) constant velocity, where
    target acceleration is modelled as Levy noise.
    """

    @property
    def constant_derivative(self):
        """For constant velocity, this is 1."""
        return 1


class LevyConstantAcceleration(LevyConstantNthDerivative):
    r"""This is a class implementation of a discrete, time-variant 1D Constant
    Acceleration Transition Model.

    The target acceleration is modeled as a Levy noise random process.
    """

    @property
    def constant_derivative(self):
        """For constant acceleration, this is 2."""
        return 2


class LevyNthDerivativeDecay(LinearLevyTransitionModel, TimeVariantModel):
    r"""Identical model to :class:`~.NthDerivativeDecay`, but driving white noise is now replaced with
    a Levy driving process.
    """
    decay_derivative: int = Property(
        doc="The derivative with respect to time to decay exponentially, eg if 2 identical to "
            "singer")
    noise_diff_coeff: Optional[float] = Property(default=1.0, doc="The acceleration noise diffusion coefficient :math:`q`")
    damping_coeff: float = Property(doc="The Nth derivative damping coefficient :math:`K`")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.A = self._init_A()
        self.h = self._init_h()
    
    def _integrand(self, dt: float, jtimes: np.ndarray) -> np.ndarray:
        delta = dt - jtimes[..., np.newaxis, np.newaxis]
        return expm(self.A * delta) @ self.h

    def _init_A(self):
        N = self.decay_derivative
        mat = np.zeros((N + 1, N + 1))
        for i in range(N):
            mat[i, i + 1] = 1
        mat[N, N] = -self.damping_coeff
        return mat

    def _init_h(self):
        N = self.decay_derivative
        mat = np.zeros((N + 1, 1))
        mat[N, 0] = 1
        return mat

    @property
    def ndim_state(self):
        return self.decay_derivative + 1


    def matrix(self, time_interval, **kwargs):
        dt = time_interval.total_seconds()
        return expm(self.A * dt)
    
    def rvs(
        self,
        latents: Optional[Latents] = None,
        num_samples: int = 1,
        random_state: RandomState = None,
        **kwargs
    ) -> Union[StateVector, StateVectors]:
        coeff = self.noise_diff_coeff
        return super().rvs(latents, num_samples, random_state, **kwargs) * coeff


class LevyLangevin(LevyNthDerivativeDecay):
    r"""This is a class implementation of a discrete, time-variant 1D
    Linear-Levy Ornstein Uhlenbeck Transition Model.

    The target is assumed to move with (nearly) constant velocity, which
    exponentially decays to zero over time, and target acceleration is
    modeled as a Levy process.
    """

    damping_coeff: float = Property(doc="The velocity damping coefficient :math:`K`")

    @property
    def decay_derivative(self):
        return 1
