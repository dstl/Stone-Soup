from abc import abstractmethod
from datetime import timedelta
import copy
from typing import Sequence, Iterable, Union, List, Optional, Callable

from scipy.linalg import block_diag
import numpy as np

from ..base_driver import Latents
from ..base import Model, GaussianModel, LinearModel, TimeVariantModel, LevyModel
from ...base import Property
from ...types.array import StateVector, StateVectors, CovarianceMatrix, CovarianceMatrices
from ...types.state import State
from ...types.numeric import Probability


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.model_list, Sequence):
            raise TypeError("model_list must be Sequence.")

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


class CombinedLevyTransitionModel(TransitionModel, LevyModel):
    r"""Combine multiple models into a single model by stacking them.

    The assumption is that all models are Gaussian.
    Time Variant, and Time Invariant models can be combined together.
    If any of the models are time variant the keyword argument "time_interval"
    must be supplied to all methods
    """
    model_list: Sequence[GaussianModel] = Property(doc="List of Transition Models.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert(len(self.model_list) != 0)

    def _integrand(self, dt: float, jtimes: np.ndarray):
        return NotImplementedError

    @property
    def driver(self) -> List[Iterable]:
        return [model.driver for model in self.model_list]

    @property
    def mu_W(self):
        mu = [m.mu_W if m.mu_W is not None else m.driver.mu_W for m in self.model_list]
        return np.atleast_2d(mu).T
    
    @property
    def sigma_W2(self):
        sigma2 = [m.sigma_W2 if m.sigma_W2 is not None else m.driver.sigma_W2 for m in self.model_list]
        return np.diag(sigma2)
    
    @property
    def ndim_state(self):
        """ndim_state getter method

        Returns
        -------
        : :class:`int`
            The number of combined model state dimensions.
        """
        return sum(model.ndim_state for model in self.model_list)


    def mean(self, **kwargs) -> Union[StateVector, StateVectors]:
        """Returns the transition model noise mean matrix.

        Returns
        -------
        : :class:`stonesoup.types.state.StateVector` of shape\
        (:py:attr:`~ndim_state`, 1)
            The process noise mean.
        """
        mean_list = [model.mean(**kwargs) for _, model in enumerate(self.model_list)]
        if len(mean_list[0].shape) == 2:
            return np.vstack(mean_list).view(StateVector)
        else:
            return np.concatenate(mean_list, axis=1).view(StateVectors)
        
    
    def covar(self, **kwargs) -> Union[CovarianceMatrix, CovarianceMatrices]:
        """Returns the transition model noise covariance matrix.

        Returns
        -------
        : :class:`stonesoup.types.state.CovarianceMatrix` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
            The process noise covariance.
        """

        covar_list = [model.covar(**kwargs) for _, model in enumerate(self.model_list)]
        if len(covar_list[0].shape) == 2:
            return block_diag(*covar_list).view(CovarianceMatrix)
        else:
            N = covar_list[0].shape[0]
            ret = []
            for n in range(N):
                tmp = []
                for tensor in covar_list: # D
                    tmp.append(tensor[n])
                ret.append(block_diag(*tmp))
            return np.array(ret).view(CovarianceMatrices)


    def function(self, state, time_interval: timedelta, noise=False, **kwargs) -> StateVector:
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
        if state.state_vector.shape[1] == 1:
            state_vector = np.zeros(state.state_vector.shape).view(StateVector)
        else:
            state_vector = np.zeros(state.state_vector.shape).view(StateVectors)
        # To handle explicit noise vector(s) passed in we set the noise for the individual models
        # to False and add the noise later. When noise is Boolean, we just pass in that value.
        if noise is None:
            noise = False
        if isinstance(noise, bool):
            noise_loop = noise
        else:
            noise_loop = False
        latents = self.sample_latents(time_interval=time_interval, num_samples=1)
        for model in self.model_list:
            temp_state.state_vector = state.state_vector[
                ndim_count : model.ndim_state + ndim_count, :
            ]
            state_vector[ndim_count : model.ndim_state + ndim_count, :] += model.function(
                state=temp_state, latents=latents, time_interval=time_interval, noise=noise_loop, **kwargs
            )
            ndim_count += model.ndim_state

        if isinstance(noise, bool):
            noise = 0
        return state_vector + noise

    def sample_latents(self, time_interval: timedelta, num_samples: int, random_state: Optional[np.random.RandomState]=None) -> Latents:
        dt = time_interval.total_seconds()
        latents = Latents(num_samples=num_samples)
        for m in self.model_list:
            if m.driver and not latents.exists(m.driver):
                jsizes, jtimes = m.driver.sample_latents(dt=dt, num_samples=num_samples, random_state=random_state)
                latents.add(driver=m.driver, jsizes=jsizes, jtimes=jtimes)
        return latents

