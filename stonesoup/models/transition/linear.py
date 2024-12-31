import math
from datetime import timedelta
from collections.abc import Sequence
from functools import lru_cache
from abc import abstractmethod
from typing import Optional

import numpy as np
from scipy.integrate import quad
from scipy.linalg import block_diag, solve
from scipy.stats import norm

from .base import TransitionModel, CombinedGaussianTransitionModel
from ..base import (LinearModel, GaussianModel, TimeVariantModel,
                    TimeInvariantModel)
from ...base import Property
from ...types.array import CovarianceMatrix


class LinearGaussianTransitionModel(
        TransitionModel, LinearModel, GaussianModel):

    @property
    def ndim_state(self):
        """ndim_state getter method

        Returns
        -------
        : :class:`int`
            The number of model state dimensions.
        """

        return self.matrix().shape[0]


class CombinedLinearGaussianTransitionModel(LinearModel, CombinedGaussianTransitionModel):
    r"""Combine multiple models into a single model by stacking them.

    The assumption is that all models are Linear and Gaussian.
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

        transition_matrices = [
            model.matrix(**kwargs) for model in self.model_list]
        return block_diag(*transition_matrices)


class LinearGaussianTimeInvariantTransitionModel(LinearGaussianTransitionModel,
                                                 TimeInvariantModel):
    r"""Generic Linear Gaussian Time Invariant Transition Model."""

    transition_matrix: np.ndarray = Property(doc="Transition matrix :math:`\\mathbf{F}`.")
    control_matrix: np.ndarray = Property(
        default=None, doc="Control matrix :math:`\\mathbf{B}`.")
    covariance_matrix: CovarianceMatrix = Property(
        doc="Transition noise covariance matrix :math:`\\mathbf{Q}`.")

    def matrix(self, **kwargs):
        """Model matrix :math:`F`

        Returns
        -------
        : :class:`numpy.ndarray` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
            The model matrix evaluated given the provided time interval.
        """

        return self.transition_matrix

    def covar(self, **kwargs):
        """Returns the transition model noise covariance matrix.

        Returns
        -------
        : :class:`stonesoup.types.state.CovarianceMatrix` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
            The process noise covariance.
        """

        return self.covariance_matrix


class ConstantNthDerivative(LinearGaussianTransitionModel, TimeVariantModel):
    r"""Discrete model based on the Nth derivative with respect to time being
    constant, to set derivative use keyword argument
    :attr:`constant_derivative`

     The model is described by the following SDEs:

        .. math::
            :nowrap:

            \begin{eqnarray}
                dx^{(N-1)} & = & x^{(N)} dt & | {(N-1)th \ derivative \ on \
                X-axis (m)} \\
                dx^{(N)} & = & q\cdot dW_t,\ W_t \sim \mathcal{N}(0,q^2) & | \
                Nth\ derivative\ on\ X-axis (m/s^{N})
            \end{eqnarray}

    It is hard to represent the matrix form of these due to the fact that they
    vary with N, examples for N=1 and N=2 can be found in the
    :class:`~.ConstantVelocity` and :class:`~.ConstantAcceleration` models
    respectively. To aid visualisation of :math:`F_t` the elements are
    calculated as the terms of the taylor expansion of each state variable.
    """

    constant_derivative: int = Property(
        doc="The order of the derivative with respect to time to be kept constant, eg if 2 "
            "identical to constant acceleration")
    noise_diff_coeff: float = Property(
        doc="The Nth derivative noise diffusion coefficient (Variance) :math:`q`")

    @property
    def ndim_state(self):
        return self.constant_derivative + 1

    def matrix(self, time_interval, **kwargs):
        """Model matrix :math:`F`

        Returns
        -------
        : :class:`numpy.ndarray` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
        """
        time_interval_sec = time_interval.total_seconds()
        N = self.constant_derivative
        Fmat = np.zeros((N + 1, N + 1))
        dt = time_interval_sec
        for i in range(0, N + 1):
            for j in range(i, N + 1):
                Fmat[i, j] = (dt ** (j - i)) / math.factorial(j - i)

        return Fmat

    def covar(self, time_interval, **kwargs):
        time_interval_sec = time_interval.total_seconds()
        dt = time_interval_sec
        N = self.constant_derivative
        if N == 1:
            covar = np.array([[dt**3 / 3, dt**2 / 2],
                              [dt**2 / 2, dt]])
        else:
            Fmat = self.matrix(time_interval, **kwargs)
            Q = np.zeros((N + 1, N + 1))
            Q[N, N] = 1
            igrand = Fmat @ Q @ Fmat.T
            covar = np.zeros((N + 1, N + 1))
            for l in range(0, N + 1):  # noqa: E741
                for k in range(0, N + 1):
                    covar[l, k] = (igrand[l, k]*dt / (1 + N*2 - l - k))
        covar *= self.noise_diff_coeff
        return CovarianceMatrix(covar)


class RandomWalk(ConstantNthDerivative):
    r"""This is a class implementation of a discrete, time-variant 1D
    Linear-Gaussian Random Walk Transition Model.

        The target is assumed to be (almost) stationary, where
        target velocity is modelled as white noise.
        """
    noise_diff_coeff: float = Property(doc="The position noise diffusion coefficient :math:`q`")

    @property
    def constant_derivative(self):
        """For random walk, this is 0."""
        return 0


class ConstantVelocity(ConstantNthDerivative):
    r"""This is a class implementation of a discrete, time-variant 1D
    Linear-Gaussian Constant Velocity Transition Model.

    The target is assumed to move with (nearly) constant velocity, where
    target acceleration is modelled as white noise.

    The model is described by the following SDEs:

        .. math::
            :nowrap:

            \begin{eqnarray}
                dx_{pos} & = & x_{vel} d & | {Position \ on \
                X-axis (m)} \\
                dx_{vel} & = & q\cdot dW_t,\ W_t \sim \mathcal{N}(0,q^2) & | \
                Speed on\ X-axis (m/s)
            \end{eqnarray}

    Or equivalently:

        .. math::
            x_t = F_t x_{t-1} + w_t,\ w_t \sim \mathcal{N}(0,Q_t)

    where:

        .. math::
            x & = & \begin{bmatrix}
                        x_{pos} \\
                        x_{vel}
                \end{bmatrix}

        .. math::
            F_t & = & \begin{bmatrix}
                        1 & dt\\
                        0 & 1
                \end{bmatrix}

        .. math::
            Q_t & = & \begin{bmatrix}
                        \frac{dt^3}{3} & \frac{dt^2}{2} \\
                        \frac{dt^2}{2} & dt
                \end{bmatrix} q
    """
    noise_diff_coeff: float = Property(doc="The velocity noise diffusion coefficient :math:`q`")

    @property
    def constant_derivative(self):
        """For constant velocity, this is 1."""
        return 1


class ConstantAcceleration(ConstantNthDerivative):
    r"""This is a class implementation of a discrete, time-variant 1D Constant
    Acceleration Transition Model.

    The target acceleration is modeled as a zero-mean white noise random
    process.

    The model is described by the following SDEs:

        .. math::
            :nowrap:

            \begin{eqnarray}
                dx_{pos} & = & x_{vel} d & | {Position \ on \
                X-axis (m)} \\
                dx_{vel} & = & x_{acc} d & | {Speed \
                on\ X-axis (m/s)} \\
                dx_{acc} & = & q W_t,\ W_t \sim
                \mathcal{N}(0,q^2) & | {Acceleration \ on \ X-axis (m^2/s)}

            \end{eqnarray}

    Or equivalently:

        .. math::
            x_t = F_t x_{t-1} + w_t,\ w_t \sim \mathcal{N}(0,Q_t)

    where:

        .. math::
            x & = & \begin{bmatrix}
                         x_{pos} \\
                         x_{vel} \\
                         x_{acc}
                    \end{bmatrix}

        .. math::
            F_t & = & \begin{bmatrix}
                           1 & dt & \frac{dt^2}{2} \\
                           0 & 1 & dt \\
                           0 & 0 & 1
                      \end{bmatrix}

        .. math::
            Q_t & = & \begin{bmatrix}
                        \frac{dt^5}{20} & \frac{dt^4}{8} & \frac{dt^3}{6} \\
                        \frac{dt^4}{8} & \frac{dt^3}{3} & \frac{dt^2}{2} \\
                        \frac{dt^3}{6} & \frac{dt^2}{2} & dt
                      \end{bmatrix} q
    """
    noise_diff_coeff: float = Property(
        doc="The acceleration noise diffusion coefficient :math:`q`")

    @property
    def constant_derivative(self):
        """For constant acceleration, this is 2."""
        return 2


class NthDerivativeDecay(LinearGaussianTransitionModel, TimeVariantModel):
    r"""Discrete model based on the Nth derivative with respect to time
    decaying to 0 exponentially, to set derivative use keyword argument
    :attr:`decay_derivative`

        The model is described by the following SDEs:

        .. math::
            :nowrap:

            \begin{eqnarray}
                dx^{(N-1)} & = & x^{(N)} dt & | {(N-1)th derivative \ on \
                X-axis (m)} \\
                dx^{(N)} & = & -K x^{N} dt + q\cdot dW_t,\ W_t \sim
                \mathcal{N}(0,q^2) & | \ Nth\ derivative\ on\ X-axis (m/s^{N})
            \end{eqnarray}

    The transition and covariance matrices are very difficult to express
    simply, but examples for N=1 and N=2 are given in
    :class:`~.OrnsteinUhlenbeck` and :class:`~.Singer` respectively.
        """
    decay_derivative: int = Property(
        doc="The derivative with respect to time to decay exponentially, eg if 2 identical to "
            "singer")
    noise_diff_coeff: float = Property(doc="The noise diffusion coefficient :math:`q`")
    damping_coeff: float = Property(doc="The Nth derivative damping coefficient :math:`K`")

    @property
    def ndim_state(self):
        return self.decay_derivative + 1

    @staticmethod
    @lru_cache()
    def _continoustransitionmatrix(t, N, K):
        FCont = np.zeros((N + 1, N + 1))
        for i in range(0, N + 1):
            FCont[i, N] = np.exp(-K * t) * (-1) ** (N - i) / K ** (N - i)
            for n in range(1, N - i + 1):
                FCont[i, N] -= (-1) ** n * t ** (N - i - n) /\
                               (math.factorial(N - i - n) * K ** n)
            for j in range(i, N):
                FCont[i, j] = (t ** (j - i)) / math.factorial(j - i)
        return FCont

    def matrix(self, time_interval, **kwargs):
        dt = time_interval.total_seconds()
        N = self.decay_derivative
        K = self.damping_coeff
        return self._continoustransitionmatrix(dt, N, K)

    @classmethod
    def _continouscovar(cls, t, N, K, k, l):  # noqa: E741
        FcCont = cls._continoustransitionmatrix(t, N, K)
        Q = np.zeros((N + 1, N + 1))
        Q[N, N] = 1
        CovarCont = FcCont @ Q @ FcCont.T
        return CovarCont[k, l]

    @classmethod
    @lru_cache()
    def _covardiscrete(cls, N, q, K, dt):
        covar = np.zeros((N + 1, N + 1))
        for k in range(0, N + 1):
            for l in range(0, N + 1):  # noqa: E741
                covar[k, l] = quad(cls._continouscovar, 0,
                                   dt, args=(N, K, k, l))[0]
        return covar * q

    def covar(self, time_interval, **kwargs):
        N = self.decay_derivative
        q = self.noise_diff_coeff
        K = self.damping_coeff
        dt = time_interval.total_seconds()
        return self._covardiscrete(N, q, K, dt)


class OrnsteinUhlenbeck(NthDerivativeDecay):
    r"""This is a class implementation of a discrete, time-variant 1D
    Linear-Gaussian Ornstein Uhlenbeck Transition Model.

    The target is assumed to move with (nearly) constant velocity, which
    exponentially decays to zero over time, and target acceleration is
    modeled as white noise.

    The model is described by the following SDEs:

        .. math::
            :nowrap:

            \begin{eqnarray}
                dx_{pos} & = & x_{vel} dt & | {Position \ on \
                X-axis (m)} \\
                dx_{vel} & = & -K x_{vel} dt + q dW_t,
                W_t \sim \mathcal{N}(0,q) & | {Speed\ on \
                X-axis (m/s)}
            \end{eqnarray}

    Or equivalently:

        .. math::
            x_t = F_t x_{t-1} + w_t,\ w_t \sim \mathcal{N}(0,Q_t)

    where:

        .. math::
            x & = & \begin{bmatrix}
                        x_{pos} \\
                        x_{vel}
                \end{bmatrix}

        .. math::
            F_t & = & \begin{bmatrix}
                        1 & \frac{1}{K}(1 - e^{-Kdt})\\
                        0 & e^{-Kdt}
                \end{bmatrix}

        .. math::
            Q_t & = & \begin{bmatrix}
                        \frac{dt - \frac{2}{K}(1 - e^{-Kdt})
                              + \frac{1}{2K}(1 - e^{-2Kdt})}{K^2} &
                        \frac{\frac{1}{K}(1 - e^{-Kdt})
                              - \frac{1}{2K}(1 - e^{-2Kdt})}{K} \\
                        \frac{\frac{1}{K}(1 - e^{-Kdt})
                              - \frac{1}{2K}(1 - e^{-2Kdt})}{K} &
                        \frac{1 - e^{-2Kdt}}{2K}
                \end{bmatrix} q
    """

    noise_diff_coeff: float = Property(doc="The velocity noise diffusion coefficient :math:`q`")
    damping_coeff: float = Property(doc="The velocity damping coefficient :math:`K`")

    @property
    def decay_derivative(self):
        return 1


class Singer(NthDerivativeDecay):
    r"""This is a class implementation of a discrete, time-variant 1D Singer
    Transition Model.

    The target acceleration is modeled as a zero-mean Gauss-Markov random
    process.

    The model is described by the following SDEs:

        .. math::
            :nowrap:

            \begin{eqnarray}
                dx_{pos} & = & x_{vel} dt & | {Position \ on \
                X-axis (m)} \\
                dx_{vel} & = & x_{acc} dt & | {Speed \
                on\ X-axis (m/s)} \\
                dx_{acc} & = & -K x_{acc} dt + q W_t,\ W_t \sim
                \mathcal{N}(0,q^2) & | {Acceleration \ on \ X-axis (m^2/s)}

            \end{eqnarray}

    Or equivalently:

        .. math::
            x_t = F_t x_{t-1} + w_t,\ w_t \sim \mathcal{N}(0,Q_t)

    where:

        .. math::
            x & = & \begin{bmatrix}
                        x_{pos} \\
                        x_{vel} \\
                        x_{acc}
                    \end{bmatrix}

        .. math::
            F_t & = & \begin{bmatrix}
                        1 & dt & (K dt-1+e^{-K dt})/K^2 \\
                        0 & 1 & (1-e^{-K dt})/K \\
                        0 & 0 & e^{-K t}
                      \end{bmatrix}

        .. math::
            Q_t & = & q \begin{bmatrix}
                    \frac{[1-e^{-2K dt}] + 2K dt +
                    \frac{2K^3 dt^3}{3}- 2K^2 dt^2 -
                    4K dt e^{-K dt} }{2K^5} &
                    \frac{(K dt - [1-e^{-K dt}])^2}{2K^4} &
                    \frac{[1-e^{-2K dt}]-2K dt e^{-K dt}}
                    {2K^3} \\
                    \frac{(K dt - [1 - e^{-K dt}])^2}{2K^4} &
                    \frac{2K dt - 4[1-e^{-K dt}] +
                    [1-e^{-2K dt}]}{2K^3} &
                    \frac{[1-e^{-K dt}]^2}{2K^2} \\
                    \frac{[1- e^{-2K dt}]-2K dt e^{-K dt}}
                    {2K^3} &
                    \frac{[1-e^{-K dt}]^2}{2K^2} &
                    \frac{1-e^{-2K dt}}{2K}
                    \end{bmatrix}
    """

    noise_diff_coeff: float = Property(
        doc="The acceleration noise diffusion coefficient :math:`q`")
    damping_coeff: float = Property(doc=r"The reciprocal of the decorrelation time :math:`\alpha`")

    @property
    def decay_derivative(self):
        return 2


class SingerApproximate(Singer):

    @property
    def decay_derivative(self):
        return 2
    r"""This is a class implementation of a discrete, time-variant 1D Singer
    Transition Model, with covariance approximation applicable for smaller time
    intervals.

    The target acceleration is modeled as a zero-mean Gauss-Markov random
    process.

    The model is described by the following SDEs:

        .. math::
            :nowrap:

            \begin{eqnarray}
                dx_{pos} & = & x_{vel} d & | {Position \ on \
                X-axis (m)} \\
                dx_{vel} & = & x_{acc} d & | {Speed \
                on\ X-axis (m/s)} \\
                dx_{acc} & = & -\alpha x_{acc} d + q W_t,\ W_t \sim
                \mathcal{N}(0,q^2) & | {Acceleration \ on \ X-axis (m^2/s)}

            \end{eqnarray}

    Or equivalently:

        .. math::
            x_t = F_t x_{t-1} + w_t,\ w_t \sim \mathcal{N}(0,Q_t)

    where:

        .. math::
            x & = & \begin{bmatrix}
                        x_{pos} \\
                        x_{vel} \\
                        x_{acc}
                    \end{bmatrix}

        .. math::
            F_t & = & \begin{bmatrix}
                        1 & dt & (\alpha dt-1+e^{-\alpha dt})/\alpha^2 \\
                        0 & 1 & (1-e^{-\alpha dt})/\alpha \\
                        0 & 0 & e^{-\alpha t}
                      \end{bmatrix}

        For small dt:

        .. math::
            Q_t & = & q \begin{bmatrix}
                        \frac{dt^5}{20} & \frac{dt^4}{8} & \frac{dt^3}{6} \\
                        \frac{dt^4}{8} & \frac{dt^3}{3} & \frac{dt^2}{2} \\
                        \frac{dt^3}{6} & \frac{dt^2}{2} & dt
                        \end{bmatrix}
    """
    def covar(self, time_interval, **kwargs):
        """Returns the transition model noise covariance matrix.

        Parameters
        ----------
        time_interval : :class:`datetime.timedelta`
            A time interval :math:`dt`

        Returns
        -------
        :class:`stonesoup.types.state.CovarianceMatrix` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
            The process noise covariance.
        """

        # time_interval_threshold is currently set arbitrarily.
        time_interval_sec = time_interval.total_seconds()

        # Only leading terms get calculated for speed.
        covar = np.array(
            [[time_interval_sec**5 / 20,
              time_interval_sec**4 / 8,
              time_interval_sec**3 / 6],
             [time_interval_sec**4 / 8,
              time_interval_sec**3 / 3,
              time_interval_sec**2 / 2],
             [time_interval_sec**3 / 6,
              time_interval_sec**2 / 2,
              time_interval_sec]]
        ) * self.noise_diff_coeff

        return CovarianceMatrix(covar)

class SlidingWindowGP(LinearGaussianTransitionModel, TimeVariantModel):
    r"""Discrete model implementing a sliding window zero-mean Gaussian process (GP).

    This model defines a GP over a sliding window of states.
    The window size is set using the :attr:`window_size` parameter. A GP at
    discrete timesteps is defined by a time vector spanning from the prediction
    time :math:`t` backward to :math:`t - L + 1`. The states in this window
    form a multivariate Gaussian distribution:

        .. math::
            p(\mathbf{x}_{t:t-L+1}) \sim \mathcal{N}(\mathbf{0}, \mathbf{K})

    For prediction, the model computes the conditional Gaussian distribution
    for the current state :math:`x_t` given the previous states
    :math:`\mathbf{x}_{t-1:t-L+1}` with time vector :math:`\mathbf{t}_{L-1}`:

        .. math::
            p(x_t | \mathbf{x}_{t-1:t-L+1}) \sim \mathcal{N}(\mu_t, \sigma^2_t)

    where:

        .. math::
            \mu_t = \mathbf{k}_{t, \mathbf{t}_{L-1}}
                    \mathbf{K}_{\mathbf{t}_{L-1}, \mathbf{t}_{L-1}}^{-1}
                    \mathbf{x}_{\mathbf{t}_{L-1}}

            \sigma^2_t = k_{t,t} - \mathbf{k}_{t, \mathbf{t}_{L-1}}
                        \mathbf{K}_{\mathbf{t}_{L-1}, \mathbf{t}_{L-1}}^{-1}
                        \mathbf{k}_{t, \mathbf{t}_{L-1}}^\top

    The state transition equation is:

        .. math::
            \mathbf{x}_t = \mathbf{F}\mathbf{x}_{t-1} + w_t,
            \quad w_t \sim \mathcal{N}(0, \mathbf{Q}),

    where:

        .. math::
            \mathbf{x}_t & = & \begin{bmatrix}
                        x_t \\
                        x_{t-1} \\
                        \vdots \\
                        x_{t-L+1}
                    \end{bmatrix}

        .. math::
            \mathbf{F} =
            \begin{bmatrix}
            \mathbf{k}_{t, \mathbf{t}_{L-1}}^\top
            \mathbf{K}_{\mathbf{t}_{L-1}, \mathbf{t}_{L-1}}^{-1} \\
            \mathbf{I}_{L-1} \quad \mathbf{0}_{L-1, 1}
            \end{bmatrix}

        .. math::
            \mathbf{Q} =
            \begin{bmatrix}
            \sigma_t^2 & \mathbf{0}_{1, L-1} \\
            \mathbf{0}_{L-1, 1} & \mathbf{0}_{L-1, L-1}
            \end{bmatrix}

    The model assumes a constant time interval between observations. To
    construct the covariance matrices, a time vector is created based on the
    current prediction timestep (:attr:`pred_time`) and the specified
    :attr:`time_interval`. The time vector spans backward over the sliding
    window with a total length of :attr:`window_size`.

    Pad the state vector with zeros if the prediction time is smaller than time_interval * window_size
    """

    window_size: int = Property(doc="Size of the sliding window :math:`L`")
    markov_approx: int = Property(doc="Order of Markov Approximation. 1 or 2", default=1)
    epsilon: float = Property(doc="Small constant added to diagonal of covariance matrix for numerical stability", default=1e-6)

    @property
    def ndim_state(self):
        return self.window_size
    
    @property
    def requires_track(self):
        return True

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

        # default to RBF kernel (to be implemented in stonesoup.kernel)?
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

        if self.markov_approx == 2:
            Fmat[0, len(f)] = 1 - f.sum()

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
        
        C = self.kernel(t, t)
        C += np.eye(np.shape(C)[0]) * self.epsilon
        f = solve(C[1:, 1:], C[1:, 0])
        noise_var = C[0, 0] - C[0, 1:] @ f
        covar = np.zeros((d, d))
        covar[0, 0] = 1
        return CovarianceMatrix(covar * noise_var)
    
    def _get_time_vector(self, track, time_interval):
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
        dt = time_interval.total_seconds()
        start_time = track.states[0].timestamp

        if self.markov_approx == 1:
            prediction_time = track.states[-1].timestamp + time_interval
            time_vector = np.array([(prediction_time - start_time).total_seconds()])
            for i in range(0, d):
                state_time = track.states[-1 - i].timestamp
                time_vector = np.append(time_vector, (state_time - start_time).total_seconds())
            return time_vector.reshape(-1, 1)

        elif self.markov_approx == 2:
            if len(track.states) < self.window_size:
                # include prior at t = 0
                return np.arange(d * dt, -dt, -dt).reshape(-1, 1)
            else:
                return np.arange(d, 0, -1).reshape(-1, 1)


class IntegratedSlidingWindowGaussianProcess(SlidingWindowGaussianProcess):
    r"""Discrete time-variant 1D Sliding Window Gaussian Process (GP) model 
    with integration, where the state :math:`x_t` models the integral of a 
    zero-mean GP :math:`g(t)`.

    The relationship between :math:`x_t` and :math:`g(t)` is defined as:

        .. math::
            x_t = x_0 + \int_{0}^{t} g(\tau) \, d\tau

    where :math:`g(t)` is a zero-mean GP with a radial basis function (RBF)
    kernel:

        .. math::
            K_g(t, t') = \sigma^2 \exp \left(-\frac{(t - t')^2}{2 \ell^2} \right)

    The output variance :math:`\sigma^2` and length scale :math:`\ell` of
    :math:`g(t)` are set with the keyword arguments :attr:`output_var` and
    :attr:`length_scale` respectively.

    The initial condition :math:`x_0` is modeled as a Gaussian random variable,
    assumed independent of :math:`g(t)`:

        .. math::
            x_0 \sim \mathcal{N}(\mu_x, \sigma^2_x)

    To approximate the integral over a sliding window of size :math:`L`,
    spanning :math:`t_L = L \cdot \Delta t` in absolute time (:math:`\Delta t`
    is the constant time interval between observations), the process is
    reformulated as:

        .. math::
            x_t = x_{t - t_L} + \int_{t - t_L}^{t} g(\tau) \, d\tau

        .. math::
            \approx x_{t - t_L} + \int_{0}^{t_L} g(\tau) \, d\tau

    The integral limits are approximated as spanning from :math:`0` to :math:`t_L`,
    assuming that the contributions from separate windows are independent.
    This approximation is introduced as the analytical derivation of the kernel
    for :math:`z(t)` requires integration limits with a fixed starting point :math:`t=0`.

    The prior :math:`x_{t - t_L}` is updated to:

        .. math::
            x_{t - t_L} \sim \mathcal{N}(\mu_{x}, \sigma^2_{t - t_L})

    To model :math:`x_t` as a zero-mean GP compatible with the state space
    formulation of the sliding window GP, the prior :math:`x_{t - t_L}` is
    first set to zero, and its effects on the mean and covariance function
    of :math:`x_t` are considered separately.
    
    :math:`x_{t - t_L}` adds a constant offset :math:`\mu_{x}` to :math:`x_t`.
    The statevector is augmented with :math:`\mu_{x}` to be included in the
    observation model. :math:`x_{t - t_L}` also adds an extra term in the
    covariance function :math:`K_x(t, t')`:

        .. math::
            K_x(t, t') = \sigma^2_{t - t_L} +
            \int_{0}^{t} \int_{0}^{t'} K_g(\tau, \tau') \, d\tau \, d\tau'

    The state transition equation for :math:`x_t` is defined as:

        .. math::
            \mathbf{x}_t = \mathbf{F}\mathbf{x}_{t-1} + w_t,
            \quad w_t \sim \mathcal{N}(0, \mathbf{Q}),

    where:

        .. math::
            \mathbf{x}_t & = & \begin{bmatrix}
                        x_t \\
                        x_{t-1} \\
                        \vdots \\
                        x_{t-L+1} \\
                        \mu_{t-L}
                    \end{bmatrix}

        .. math::
            \mathbf{F} =
            \begin{bmatrix}
            \mathbf{F}' & \mathbf{0}_{1, L} \\
            \mathbf{0}_{L, 1} & \mathbf{0}_{1}
            \end{bmatrix}

        .. math::
            \mathbf{Q} =
            \begin{bmatrix}
            \sigma_t^2 & \mathbf{0}_{1, L} \\
            \mathbf{0}_{L, 1} & \mathbf{0}_{L, L}
            \end{bmatrix}

    where :math:`\mathbf{F}'` and :math:`\sigma_t^2` are computed the same
    way as the sliding window GP model but with covariance function
    :math:`K_x(t, t')`.

    The model assumes a constant time interval between observations. To
    construct the covariance matrices, a time vector is created based on the
    current prediction timestep (:attr:`pred_time`) and the specified
    :attr:`time_interval`. The time vector spans backward over the sliding
    window with a total length of :attr:`window_size`.

    :attr:`pred_time` must be supplied to all methods in this model and
    represents the elapsed duration since the start time (i.e., the time
    of the initial state).

    References
    ----------
    For a full derivation of the integral and implementation details, see the
    accompanying paper and documentation.
    """

    kernel_length_scale: float = Property(doc="RBF Kernel length scale parameter :math:`\ell`")
    kernel_output_variance: float = Property(doc="RBF kernel output variance parameter :math:`\sigma^2`")
    prior_var: float = Property(doc="Variance :math:`\sigma^2_x` of Gaussian initial value :math:`x_0`")

    @property
    def ndim_state(self):
        return self.window_size + 1

    def kernel(self, x, y, **kwargs):
        l = self.kernel_length_scale
        var = self.kernel_output_variance
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        prior_var = kwargs.get('prior_var', self.prior_var)

        K = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                K[i, j] = self._1d_kernel(l, var, float(x[i]), float(y[j])) + prior_var
        return K

    @staticmethod
    @lru_cache
    def _1d_kernel(l, var, x, y):
        sum_term = x * (norm.cdf(x / l) - norm.cdf((x - y) / l)) \
                   + y * (norm.cdf(y / l) - norm.cdf((y - x) / l)) \
                   + (l ** 2) * (norm.pdf(x, scale=l) + norm.pdf(y, scale=l) 
                                 - norm.pdf(x, loc=y, scale=l)) \
                   - l / np.sqrt(2 * np.pi)
        return np.sqrt(2 * np.pi) * l * var * sum_term
    
    @lru_cache
    def _compute_current_prior_var(cls, l, var, L, t):
        num_windows = t // L  # number of full windows since t = 0
        re = t % L  # remaining number of seconds not included in a full window
        return cls.prior_var + cls._1d_kernel(l, var, re, re) + cls._1d_kernel(l, var, L, L) * num_windows

    def matrix(self, pred_time, **kwargs):
        l = self.kernel_length_scale
        var = self.kernel_output_variance
        L = self.window_size
        prior_t = pred_time.total_seconds() - L
        current_prior_var = self._compute_current_prior_var(l, var, L, prior_t)

        base_matrix = super().matrix(pred_time=timedelta(seconds=L), prior_var=current_prior_var, **kwargs)
        padded_matrix = np.pad(base_matrix, ((0, 1), (0, 1)))
        padded_matrix[-1, -1] = 1
        return padded_matrix

    def covar(self, pred_time, **kwargs):
        l = self.kernel_length_scale
        var = self.kernel_output_variance
        L = self.window_size
        prior_t = pred_time.total_seconds() - L
        current_prior_var = self._compute_current_prior_var(l, var, L, prior_t)

        base_covar = super().covar(pred_time=timedelta(seconds=L), prior_var=current_prior_var, **kwargs)
        covar = np.pad(base_covar, ((0, 1)))
        return covar


class KnownTurnRateSandwich(LinearGaussianTransitionModel, TimeVariantModel):
    r"""This is a class implementation of a time-variant 2D Constant Turn
    Model. This model is used, as opposed to the normal :class:`~.KnownTurnRate`
    model, when the turn occurs in 2 dimensions that are not adjacent in the
    state vector, eg if the turn occurs in the x-z plane but the state vector
    is of the form :math:`(x,y,z)`. The list of transition models are to be
    applied to any state variables that lie in between, eg if for the above
    example you wanted the y component to move with constant velocity, you
    would put a :class:`~.ConstantVelocity` model in the list.

    The target is assumed to move with (nearly) constant velocity and also
    known (nearly) constant turn rate.
    """

    turn_noise_diff_coeffs: np.ndarray = Property(
        doc="The acceleration noise diffusion coefficients :math:`q`")
    turn_rate: float = Property(
        doc=r"The turn rate :math:`\omega`")
    model_list: Sequence[LinearGaussianTransitionModel] = Property(
        doc="List of Transition Models.")

    @property
    def ndim_state(self):
        """ndim_state getter method

        Returns
        -------
        : :class:`int`
            The number of combined model state dimensions.
        """
        return sum(model.ndim_state for model in self.model_list)+4

    def matrix(self, time_interval, **kwargs):
        """Model matrix :math:`F`

        Returns
        -------
        : :class:`numpy.ndarray` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
        """
        time_interval_sec = time_interval.total_seconds()
        turn_ratedt = self.turn_rate * time_interval_sec
        z = np.zeros([2, 2])
        transition_matrices = [
            model.matrix(time_interval) for model in self.model_list]
        sandwich = block_diag(z, *transition_matrices, z)
        sandwich[0:2, 0:2] = np.array([[1, np.sin(turn_ratedt)/self.turn_rate],
                                      [0, np.cos(turn_ratedt)]])
        sandwich[0:2, -2:] = np.array(
            [[0, (np.cos(turn_ratedt)-1)/self.turn_rate],
             [0, -np.sin(turn_ratedt)]])
        sandwich[-2:, 0:2] = np.array(
            [[0, (1-np.cos(turn_ratedt))/self.turn_rate],
             [0, np.sin(turn_ratedt)]])
        sandwich[-2:, -2:] = np.array([[1, np.sin(turn_ratedt)/self.turn_rate],
                                       [0, np.cos(turn_ratedt)]])
        return sandwich

    def covar(self, time_interval, **kwargs):
        """Returns the transition model noise covariance matrix.

        Returns
        -------
        : :class:`stonesoup.types.state.CovarianceMatrix` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
            The process noise covariance.
        """
        q1, q2 = self.turn_noise_diff_coeffs
        dt = time_interval.total_seconds()
        covar_list = [model.covar(time_interval) for model in self.model_list]
        ctc1 = np.array([[q1*dt**3/3, q1*dt**2/2],
                         [q1*dt**2/2, q1*dt]])
        ctc2 = np.array([[q1*dt**3/3, q1*dt**2/2],
                         [q1*dt**2/2, q1*dt]])
        return CovarianceMatrix(block_diag(ctc1, *covar_list, ctc2))


class KnownTurnRate(KnownTurnRateSandwich):
    r"""This is a class implementation of a discrete, time-variant 2D Constant
    Turn Model.

    The target is assumed to move with (nearly) constant velocity and also
    known (nearly) constant turn rate.

    The model is described by the following SDEs:

        .. math::
            :nowrap:

            \begin{eqnarray}
                dx_{pos} & = & x_{vel} d & | {Position \ on \
                X-axis (m)} \\
                dx_{vel} & = &-\omega y_{pos} d & | {Speed \
                on\ X-axis (m/s)} \\
                dy_{pos} & = & y_{vel} d & | {Position \ on \
                Y-axis (m)} \\
                dy_{vel} & = & \omega x_{pos} d & | {Speed \
                on\ Y-axis (m/s)}
            \end{eqnarray}

    Or equivalently:

        .. math::
            x_t = F_t x_{t-1} + w_t,\ w_t \sim \mathcal{N}(0,Q_t)

    where:

        .. math::
            x & = & \begin{bmatrix}
                        x_{pos} \\
                        x_{vel} \\
                        y_{pos} \\
                        y_{vel}
                    \end{bmatrix}

        .. math::
            F_t & = & \begin{bmatrix}
                          1 & \frac{\sin\omega dt}{\omega} &
                          0 &-\frac{1-\cos\omega dt}{\omega} \\
                          0 & \cos\omega dt & 0 & -\sin\omega dt \\
                          0 & \frac{1-\cos\omega dt}{\omega} &
                          1 & \frac{\sin\omega dt}{\omega}\\
                          0 & \sin\omega dt & 0 & \cos\omega dt
                      \end{bmatrix}

        .. math::
            Q_t & = & \begin{bmatrix}
                          q_x \frac{dt^3}{3} & q_x \frac{dt^2}{2} &
                          0 & 0 \\
                          q_x \frac{dt^2}{2} & q_x dt &
                          0 & 0 \\
                          0 & 0 &
                          q_y \frac{dt^3}{3} & q_y \frac{dt^2}{2}\\
                          0 & 0 &
                          q_y \frac{dt^2}{2} & q_y dt
                      \end{bmatrix}
    """

    @property
    def model_list(self):
        """For a turn in adjacent state vectors,
         no transition models go in between"""
        return []
