import copy
from abc import abstractmethod
from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from scipy.linalg import block_diag, solve, expm

from ..base import Model, GaussianModel, LinearModel, TimeVariantModel
from ...base import Property
from ...types.array import StateVector, StateVectors, CovarianceMatrix


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


class SlidingWindowGP(TransitionModel, GaussianModel, LinearModel, TimeVariantModel):
    r"""Discrete model implementing a sliding window zero-mean Gaussian process (GP).
    """

    window_size: int = Property(doc="Size of the sliding window :math:`L`")
    epsilon: float = Property(
        doc="Small constant added to diagonal of covariance matrix", default=1e-6)
    kernel_params: dict = Property(doc="Dictionary containing the keyword arguments for the kernel.")

    @property
    def requires_track(self):
        return True
    
    @property
    def ndim_state(self):
        return self.window_size

    @abstractmethod
    def kernel(self, t1, t2, **kwargs) -> np.ndarray:
        """Covariance function (kernel) of the Gaussian Process.

        Computes the covariance matrix between two time vectors using a
        kernel function.

        Parameters
        ----------
        t1 : array-like, shape (n_samples_1,)
        
        t2 : array-like, shape (n_samples_2,)

        Returns
        -------
        np.ndarray, shape (n_samples_1, n_samples_2)
            The covariance matrix between the input arrays `t1` and `t2`.
            Each entry (i, j) represents the covariance between `t1[i]` and `t2[j]`.
        """
        raise NotImplementedError

    def matrix(self, track, time_interval, **kwargs):
        """Model matrix :math:`F`

        Returns
        -------
        : :class:`numpy.ndarray` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
        """
        d = self.window_size
        t = self._get_time_vector(track, time_interval)

        C = self.kernel(t, t)
        C += np.eye(np.shape(C)[0]) * self.epsilon
        f = solve(C[1:, 1:], C[1:, 0])
        Fmat = np.eye(d, k=-1)
        Fmat[0, :len(f)] = f.T

        return Fmat

    def covar(self, track, time_interval, **kwargs):
        """Returns the transition model noise covariance matrix.

        Parameters
        ----------
        track : :class:`~.Track`
            The track containing the states to obtain the time vector
        time_interval : :class:`datetime.timedelta`
            A time interval :math:`dt`

        Returns
        -------
        :class:`stonesoup.types.state.CovarianceMatrix` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
            The process noise covariance.
        """
        d = self.ndim_state
        t = self._get_time_vector(track, time_interval)
        
        C = self.kernel(t, t, **kwargs)
        C += np.eye(np.shape(C)[0]) * self.epsilon
        f = solve(C[1:, 1:], C[1:, 0])
        noise_var = C[0, 0] - C[0, 1:] @ f
        covar = np.zeros((d, d))
        covar[0, 0] = 1
        return CovarianceMatrix(covar * noise_var)
    
    def _get_time_vector(self, track, time_interval):
        return self._get_time_vector_markov1(track, time_interval)
    
    def _get_time_vector_markov1(self, track, time_interval):
        """
        Generates a time vector containing the time elapsed (in seconds)
        from the start time of the track.

        The time vector includes:
            - The prediction time: (last state timestamp + time_interval)
            - The timestamps of the last `window_size` states in the track.

        Parameters:
            track: The track containing the states.
            time_interval: The time interval for prediction.
        
        Returns:
            time_vector (ndarray): A 2D array of elapsed times (d+1 x 1).

        Markov approx 2 assumes a constatn time interval. 
        Window spans absolute time time-interval*window_size
        """
        d = min(self.window_size, len(track.states))
        start_time = track.states[0].timestamp

        prediction_time = track.states[-1].timestamp + time_interval
        time_vector = np.array([(prediction_time - start_time).total_seconds()])
        for i in range(0, d):
            state_time = track.states[-1 - i].timestamp
            time_vector = np.append(time_vector, (state_time - start_time).total_seconds())
        return time_vector.reshape(-1, 1)


class DynamicsInformedIntegratedGP(SlidingWindowGP):
    r"""Discrete time-variant 1D Dynamics Informed Integrated Gaussian Process (iDGP) model,
    where velocity is modelled as a first order DE, with a GP as driving noise.
    """

    markov_approx: int = Property(doc="Order of Markov Approximation. 1 or 2", default=1)
    dynamics_coeff: float = Property(doc="Coefficient a of equation dx/dt = ax + bg(t)")
    gp_coeff: float = Property(doc="Coefficient b of equation dx/dt = ax + bg(t)")
    prior_var: float = Property(
        doc="Variance of prior x_0. Added to covariance function during initialisation", default=0)

    @property
    def ndim_state(self):
        return self.window_size + 1 if self.markov_approx == 1 else self.window_size
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.dynamics_coeff == 0 and not isinstance(self, IntegratedGP) and not isinstance(self, TwiceIntegratedGP):
            raise ValueError("dynamics_coeff cannot be 0. Use IntegratedGP or TwiceIntegratedGP classes instead.")

    def matrix(self, track, time_interval, **kwargs):
        a = self.dynamics_coeff
        d = self.window_size
        dt = time_interval.total_seconds()

        Fmat = super().matrix(track=track, time_interval=time_interval, **kwargs)

        # Add extra dimension for augmented mean
        if self.markov_approx == 1:
            Fmat = np.pad(Fmat, ((0, 1), (0, 1)))
            Fmat[-1, -1] = np.exp(a * dt)
        else:
            # Compute term for x_{k-d} (after initialisation)
            t_d = self._get_time_vector_markov2(track, time_interval)
            if len(track.states) >= self.window_size:
                Fmat[0, -1] = np.exp(a * dt * d) - np.dot(Fmat[0, :-1], np.exp(a * t_d[1:]))

        return Fmat
    
    def _get_time_vector(self, track, time_interval):
        if self.markov_approx == 1:
            return self._get_time_vector_markov1(track, time_interval)
        return self._get_time_vector_markov2(track, time_interval)
    
    def _get_time_vector_markov2(self, track, time_interval):
        d = min(self.window_size, len(track.states))
        dt = time_interval.total_seconds()
    
        if len(track.states) < self.window_size:
            # include prior at t = 0
            return np.linspace(d * dt, 0, d + 1).reshape(-1, 1)
        else:
            return np.linspace(d * dt, dt, d).reshape(-1, 1)
    
    def kernel(self, t1, t2):
        "Computes 2D covariance matrix element-wise"
        t1 = np.atleast_1d(t1)
        t2 = np.atleast_1d(t2)

        K = np.zeros((len(t1), len(t2)))

        # if t1 == t2, compute upper triangular matrix only
        if np.array_equal(t1, t2):
            for i in range(len(t1)):
                for j in range(i, len(t2)):
                    K[i, j] = self._invoke_scalar_kernel(float(t1[i]), float(t2[j]))
                    if i != j:
                        K[j, i] = K[i, j]
        
        # Compute full matrix
        else:
            for i in range(len(t1)):
                for j in range(len(t2)):
                    K[i, j] = self._invoke_scalar_kernel(float(t1[i]), float(t2[j]))

        # Include prior variance if the current window includes the prior
        if t1[-1] == 0:
            K += self.prior_var

        return K
    
    def _invoke_scalar_kernel(self, t1, t2):
        """Unpacks kernel paramaters and passes it to kernel method to enable caching."""
        if isinstance(self, (IntegratedGP, TwiceIntegratedGP)):
            return self._scalar_kernel(t1, t2, **self.kernel_params)

        return self._scalar_kernel(
            t1, t2, dynamics_coeff=self.dynamics_coeff, gp_coeff=self.gp_coeff, **self.kernel_params)

    @staticmethod
    @abstractmethod
    def _scalar_kernel(t1, t2, **kwargs):
        raise NotImplementedError


class DynamicsInformedTwiceIntegratedGP(DynamicsInformedIntegratedGP):
    r"""Discrete time-variant 1D Dynamics Informed Twice Integrated Gaussian Process (iiDGP) model,
    where acceleration is modelled as a first order DE, with a GP as driving noise.
    This model is implemented with the first Markovian approximation only.
    """

    @property
    def markov_approx(self):
        return 1  # markov_approx = 2 not implemented

    @property
    def ndim_state(self):
        return self.window_size + 2

    def matrix(self, track, time_interval, **kwargs):
        d = self.window_size
        dt = time_interval.total_seconds()

        Fmat = np.zeros((self.ndim_state, self.ndim_state))
        Fmat[:d, :d] = super().matrix(track, time_interval, **kwargs)[:d, :d]

        A_mean = np.array([[0, 1],
                            [0, self.dynamics_coeff]])
        Fmat_mean = expm(A_mean * dt)  # 2x2 sub-transition matrix for [mean_pos, mean_vel]
        Fmat[d:, d:] = Fmat_mean
        return Fmat
    

class IntegratedGP(DynamicsInformedIntegratedGP):
    r"""Discrete time-variant 1D Integrated Gaussian Process (iGP) model,
    where velocity is modelled with a GP as driving noise.
    """

    @property
    def dynamics_coeff(self):
        return 0
    
    @property
    def gp_coeff(self):
        return 1
    

class TwiceIntegratedGP(DynamicsInformedTwiceIntegratedGP):
    r""""Discrete time-variant 1D Twice Integrated Gaussian Process (iiGP) model,
    where acceleration is modelled with a GP as driving noise.
    """

    @property
    def dynamics_coeff(self):
        return 0
    
    @property
    def gp_coeff(self):
        return 1