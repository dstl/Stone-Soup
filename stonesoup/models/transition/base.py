from abc import abstractmethod
from datetime import timedelta
import copy
from typing import Sequence, Iterable, Union, List, Optional

from scipy.linalg import block_diag
import numpy as np

from .base_driver import Latents, GaussianDriver, ConditionalGaussianDriver
from ..base import Model, GaussianModel, LinearModel
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


class DrivenTransitionModel(TransitionModel):
    g_driver: GaussianDriver = Property(default=None, doc="Gaussian noise process.")
    cg_driver: ConditionalGaussianDriver = Property(default=None, doc="Conditional Gaussian noise process.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # To prevent checks throwing errors due to self.g_driver returning a list instead
        # In addition, checks for combined model are dedundant as they are performed when each dimension is initialized individually.
        if hasattr(self, "model_list"): return 
        # if self.g_driver and self.ndim_state != self.g_driver.ndim_state:
        #     raise AttributeError("No. of state dimensions of model and Gaussian driving noise process must match.")
        # if self.cg_driver and self.ndim_state != self.cg_driver.ndim_state:
        #     raise AttributeError("No. of state dimensions of model and conditionally Gaussian driving noise process must match.")

    @abstractmethod
    def ft(self, dt: float, jtimes: np.ndarray, **kwargs) -> np.ndarray:
        """
        Returns function handle implementing f_t() = exp(At) @ h
        """
        pass

    @abstractmethod
    def e_ft(self, dt: float, **kwargs) -> np.ndarray:
        """
        Returns function handle implementing E[f_t()]
        """
        pass

    # @abstractmethod
    # def ft2(self, dt: float, jtimes2: np.ndarray, **kwargs) -> np.ndarray:
    #     """
    #     Returns function handle implementing f_t() = exp(At) @ h @ h.T @ exp(At).T
    #     """
    #     pass

    # @abstractmethod
    # def e_ft2(self, dt: float, **kwargs) -> np.ndarray:
    #     """
    #     Returns function handle implementing E[f_t() @ f_t().T]
    #     """
    #     pass

    @abstractmethod
    def e_gt(self, dt: float, **kwargs) -> np.ndarray:
        """
        Returns function handle implementing E[g_t()]
        """
        pass

    # @abstractmethod
    # def e_gt2(self, dt: float, **kwargs) -> np.ndarray:
    #     """
    #     Returns function handle implementing E[g_t() @ g_t().T]
    #     """
    #     pass
    
    def sample_latents(self, time_interval: timedelta, num_samples: int) -> Latents:
        dt = time_interval.total_seconds()
        latents = Latents(num_samples=num_samples)
        if self.cg_driver:
            jsizes, jtimes = self.cg_driver.sample_latents(dt=dt, num_samples=num_samples)
            latents.add(driver=self.cg_driver, jsizes=jsizes, jtimes=jtimes)
        return latents
    
    def mean(self, latents: Latents, time_interval: timedelta, **kwargs) -> StateVector | StateVectors:
        dt = time_interval.total_seconds()
        mean = np.zeros((self.ndim_state, 1))
        if self.g_driver:
            mean += self.g_driver.mean(e_gt_func=self.e_gt, dt=dt)
        if self.cg_driver:
            tmp = self.cg_driver.mean(latents=latents, ft_func=self.ft, e_ft_func=self.e_ft, dt=dt)
            if isinstance(tmp, StateVectors):
                n_samples = tmp.shape[0]
                mean = np.tile(mean, (n_samples, 1, 1))
            mean += tmp
        return mean
    
    def covar(self, latents: Latents, time_interval: timedelta, **kwargs) -> CovarianceMatrix | CovarianceMatrices:
        # dt = time_interval.total_seconds()
        # covar = 0
        # if self.g_driver:
        #     covar += self.g_driver.covar(e_gt2_func=self.e_gt2, dt=dt)
        # if self.cg_driver:
        #     tmp = self.cg_driver.covar(latents=latents, ft2_func=self.ft2, e_ft2_func=self.e_ft2, dt=dt)
        #     if isinstance(tmp, CovarianceMatrices):
        #         covar = covar[None, ...]
        #     covar += tmp
        # return covar
        dt = time_interval.total_seconds()
        covar = np.zeros((self.ndim_state, self.ndim_state))
        if self.g_driver:
            covar += self.g_driver.covar(e_gt_func=self.e_gt, dt=dt)
        if self.cg_driver:
            tmp = self.cg_driver.covar(latents=latents, ft_func=self.ft, e_ft_func=self.e_ft, dt=dt)
            if isinstance(tmp, CovarianceMatrices):
                n_samples = tmp.shape[0]
                covar = np.tile(covar, (n_samples, 1, 1))
            covar += tmp
        return covar

    def rvs(
        self, time_interval: timedelta, num_samples: int = 1, latents: Optional[Latents]=None, **kwargs
    ) -> StateVector | StateVectors:
        """Linear combination of Gaussian noise samples"""
        dt = time_interval.total_seconds()
        noise = 0
        if self.g_driver:
            mean = self.g_driver.mean(e_gt_func=self.e_gt, dt=dt)
            # covar = self.g_driver.covar(e_gt2_func=self.e_gt2, dt=dt)
            covar = self.g_driver.covar(e_gt_func=self.e_gt, dt=dt)
            noise += self.g_driver.rvs(mean=mean, covar=covar, num_samples=num_samples, **kwargs)

        if self.cg_driver:
            if not latents:
                latents = self.sample_latents(time_interval=time_interval, num_samples=1)
            mean = self.cg_driver.mean(latents=latents, ft_func=self.ft, e_ft_func=self.e_ft, dt=dt)
            # covar = self.cg_driver.covar(latents=latents, ft2_func=self.ft2, e_ft2_func=self.e_ft2, dt=dt)
            covar = self.cg_driver.covar(latents=latents, ft_func=self.ft, e_ft_func=self.e_ft, dt=dt)
            noise += self.cg_driver.rvs(mean=mean, covar=covar, num_samples=num_samples, **kwargs)

        return noise

    def pdf(self, *args, **kwargs) -> Union[Probability, np.ndarray]:
        raise NotImplementedError

    def logpdf(self, *args, **kwargs) -> Union[float, np.ndarray]:
        raise NotImplementedError


class LinearDrivenTransitionModel(DrivenTransitionModel, LinearModel):
    @property
    def ndim_state(self, **kwargs):
        """ndim_state getter method

        Returns
        -------
        : :class:`int`
            The number of model state dimensions.
        """
        # Time delta does not affect matrix dimensions.
        return self.matrix(time_interval=timedelta(seconds=1), **kwargs).shape[0]


class CombinedDrivenTransitionModel(DrivenTransitionModel):
    model_list: Sequence[DrivenTransitionModel] = Property(doc="List of Transition Models.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert(len(self.model_list) != 0)

    @property
    def g_driver(self) -> List[Iterable]:
        return [model.g_driver for model in self.model_list]
    
    @property
    def cg_driver(self) -> List[Iterable]:
        return [model.cg_driver for model in self.model_list]
       
    @property
    def ndim_state(self):
        """ndim_state getter method

        Returns
        -------
        : :class:`int`
            The number of combined model state dimensions.
        """
        return sum(model.ndim_state for model in self.model_list)

    def sample_latents(self, time_interval: timedelta, num_samples: int) -> Latents:
        dt = time_interval.total_seconds()
        latents = Latents(num_samples=num_samples)
        for m in self.model_list:
            if m.cg_driver and not latents.exists(m.cg_driver):
                jsizes, jtimes = m.cg_driver.sample_latents(dt=dt, num_samples=num_samples)
                latents.add(driver=m.cg_driver, jsizes=jsizes, jtimes=jtimes)
        return latents
    
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

    def ft(self, **kwargs) -> np.ndarray:
        tmp = [model.ft(**kwargs) for model in self.model_list]
        return np.hstack(tmp)

    def e_ft(self, **kwargs) -> np.ndarray:
        tmp = [model.e_ft(**kwargs) for model in self.model_list]
        return np.vstack(tmp)

    # def ft2(self, **kwargs) -> np.ndarray:
    #     tmp = [model.ft2(**kwargs) for model in self.model_list]
    #     combined = []
    #     for i in range(tmp[0].shape[0]):  # Loop through first axis
    #         matrices = []
    #         for j in range(len(tmp)):
    #             matrices.append(tmp[j][i, ...])
    #         combined.append(block_diag(*matrices))
    #     return combined

    # def e_ft2(self, **kwargs) -> np.ndarray:
    #     tmp = [model.e_ft2(**kwargs) for model in self.model_list]
    #     return block_diag(*tmp)

    def e_gt(self, **kwargs) -> np.ndarray:
        tmp = [model.e_gt(**kwargs) for model in self.model_list]
        return np.vstack(tmp)

    # def e_gt2(self, **kwargs) -> np.ndarray:
    #     tmp = [model.e_gt2(**kwargs) for model in self.model_list]
    #     return block_diag(*tmp)

    def covar(self, **kwargs) -> CovarianceMatrix | CovarianceMatrices:
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

    def mean(self, **kwargs) -> StateVector | StateVectors:
        mean_list = [model.mean(**kwargs) for _, model in enumerate(self.model_list)]
        if len(mean_list[0].shape) == 2:
            return np.vstack(mean_list).view(StateVector)
        else:
            return np.concatenate(mean_list, axis=1).view(StateVectors)
        

    def jacobian(self, state, **kwargs) -> np.ndarray:
        temp_state = copy.copy(state)
        ndim_count = 0
        J_list = []
        for model in self.model_list:
            temp_state.state_vector = state.state_vector[
                ndim_count : model.ndim_state + ndim_count, :
            ]
            J_list.append(model.jacobian(temp_state, **kwargs))

            ndim_count += model.ndim_state
        out = block_diag(*J_list)

        return out


class CombinedLinearDrivenTransitionModel(CombinedDrivenTransitionModel, LinearModel):
    def matrix(self, **kwargs):
        """Model matrix :math:`F`

        Returns
        -------
        : :class:`numpy.ndarray` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
        """

        transition_matrices = [model.matrix(**kwargs) for model in self.model_list]
        return block_diag(*transition_matrices)
