from abc import abstractmethod
import copy
from typing import Sequence

from scipy.linalg import block_diag
import numpy as np

from ..base import Model, GaussianModel
from ...base import Property
from ...types.array import StateVector, StateVectors


class TransitionModel(Model):
    """Transition Model base class"""

    @property
    def ndim(self) -> int:
        return self.ndim_state

    @property
    @abstractmethod
    def ndim_state(self) -> int:
        """Number of state dimensions"""
        pass


class CombinedGaussianTransitionModel(TransitionModel, GaussianModel):
    r"""Combine multiple models into a single model by stacking them.

    The assumption is that all models are Gaussian.
    Time Variant, and Time Invariant models can be combined together.
    If any of the models are time variant the keyword argument "time_interval"
    must be supplied to all methods
    """
    model_list: Sequence[GaussianModel] = Property(doc="List of Transition Models.")

    def function(self, state, noise=False, **kwargs) -> StateVector:
        """Applies each transition model in :py:attr:`~model_list` in turn to the state's
        corresponding state vector components.
        For example, in a 3D state space, with :py:attr:`~model_list` = [modelA(ndim_state=2),
        modelB(ndim_state=1)], this would apply modelA to the state vector's 1st and 2nd elements,
        then modelB to the remaining 3rd element.

        Parameters
        ----------
        state : :class:`stonesoup.state.State`
            The state to be transitioned according to the models in :py:attr:`~model_list`.

        Returns
        -------
        state_vector: :class:`stonesoup.types.array.StateVector`
            of shape (:py:attr:`~ndim_state, 1`). The resultant state vector of the transition.
        """
        temp_state = copy.copy(state)
        ndim_count = 0
        if state.state_vector.shape[1] > 1:
            state_vector = np.zeros(state.state_vector.shape).view(StateVectors)
        else:
            state_vector = np.zeros(state.state_vector.shape).view(StateVector)
        # To handle explicit noise vector(s) passed in we set the noise for the individual models
        # to False and add the noise later. When noise is Boolean, we just pass in that value.
        if noise is None:
            noise = False
        if isinstance(noise, bool):
            noise_loop = noise
        else:
            noise_loop = False
        for model in self.model_list:
            temp_state.state_vector =\
                state.state_vector[ndim_count:model.ndim_state + ndim_count, :]
            state_vector[ndim_count:model.ndim_state + ndim_count, :] += \
                model.function(temp_state, noise=noise_loop, **kwargs)
            ndim_count += model.ndim_state
        if isinstance(noise, bool):
            noise = 0
        return state_vector + noise

    def jacobian(self, state, **kwargs):
        """Model jacobian matrix :math:`H_{jac}`

        Parameters
        ----------
        state : :class:`~.State`
            An input state

        Returns
        -------
        :class:`numpy.ndarray` of shape (:py:attr:`~ndim_meas`, \
        :py:attr:`~ndim_state`)
            The model jacobian matrix evaluated around the given state vector.
        """
        temp_state = copy.copy(state)
        ndim_count = 0
        J_list = []
        for model in self.model_list:
            temp_state.state_vector =\
                state.state_vector[ndim_count:model.ndim_state + ndim_count, :]
            J_list.append(model.jacobian(temp_state, **kwargs))

            ndim_count += model.ndim_state
        out = block_diag(*J_list)
        return out

    @property
    def ndim_state(self):
        """ndim_state getter method

        Returns
        -------
        : :class:`int`
            The number of combined model state dimensions.
        """
        return sum(model.ndim_state for model in self.model_list)

    def covar(self, **kwargs):
        """Returns the transition model noise covariance matrix.

        Returns
        -------
        : :class:`stonesoup.types.state.CovarianceMatrix` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
            The process noise covariance.
        """

        covar_list = [model.covar(**kwargs) for model in self.model_list]
        return block_diag(*covar_list)
