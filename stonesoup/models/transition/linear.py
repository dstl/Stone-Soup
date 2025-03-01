import math
from datetime import timedelta
from collections.abc import Sequence
from functools import lru_cache
from abc import abstractmethod
from typing import Optional

import numpy as np
from scipy.integrate import quad
from scipy.linalg import block_diag
from scipy.stats import norm
from scipy.special import erf

from .base import (TransitionModel, CombinedGaussianTransitionModel,
                   SlidingWindowGP, IntegratedGP, TwiceIntegratedGP,
                   DynamicsInformedIntegratedGP,
                   DynamicsInformedTwiceIntegratedGP)
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


class SlidingWindowGPSE(SlidingWindowGP):

    def kernel(self, t1, t2):
        l = self.kernel_params["length_scale"]
        var = self.kernel_params["kernel_variance"]
        t1 = t1.reshape(-1, 1)
        t2 = t2.reshape(1, -1)
        return var * np.exp(-0.5 * ((t1 - t2)/l) ** 2)


class IntegratedGPSE(IntegratedGP):
    @staticmethod
    @lru_cache
    def _scalar_kernel(t1, t2, length_scale, kernel_variance):
        l = length_scale
        var = kernel_variance
        return np.sqrt(2 * np.pi) * l * var * (
                    IntegratedGPSE._h(l, t1, 0) + IntegratedGPSE._h(l, 0, t2)
                    - IntegratedGPSE._h(l, t1, t2) - 2 / np.sqrt(2 * np.pi)
                    ) 

    @staticmethod
    @lru_cache
    def _h(l, t1, t2):
        """Helper function for _scalar_kernel."""        
        return (t1 - t2) * norm.cdf((t1 - t2) / l) + l**2 * norm.pdf(t1, loc=t2, scale=l)


class DynamicsInformedIntegratedGPSE(DynamicsInformedIntegratedGP):

    @staticmethod
    @lru_cache
    def _scalar_kernel(t1, t2, dynamics_coeff, gp_coeff, length_scale, kernel_variance):
        # length_scale and kernel_variance are from the dictionary self.kernel_params
        # the dict is unpacked in _invoke_kernel in DynamicsInformedIntegratedGP
        a = dynamics_coeff
        b = gp_coeff
        l = length_scale
        var = kernel_variance
        return (b ** 2) * var * np.sqrt(np.pi / 2) * l * (
                    DynamicsInformedIntegratedGPSE._h(a, l, t2, t1)
                    + DynamicsInformedIntegratedGPSE._h(a, l, t1, t2)
                )

    @staticmethod
    @lru_cache
    def _h(a, l, t1, t2):
        """Helper function for _dynamics_informed_kernel."""
        l_s = l * np.sqrt(2)
        gma = -a * l_s / 2
        t1_s = t1 / l_s
        t2_s = t2 / l_s
        diff_s = (t1 - t2) / l_s
        
        return ((np.exp(gma ** 2)) / (-2 * a)) * (
            np.exp(a * (t1 - t2)) * (erf(diff_s - gma) + erf(t2_s + gma))
            - np.exp(a * (t1 + t2)) * (erf(t1_s - gma) + erf(gma))
        )

class TwiceIntegratedGPSE(TwiceIntegratedGP):
    
    @staticmethod
    @lru_cache
    def _scalar_kernel(t1, t2, length_scale, kernel_variance):
        l = length_scale
        var = kernel_variance
        h = IntegratedGPSE._h
        h2 = TwiceIntegratedGPSE._h2
    
        s1 = 0.5 * t2 * h2(l, t1) - 0.5 * t1 * h2(l, -t2)

        s2 = (1 / 6) * ((t1 - t2) ** 2 * h(l, t1, t2) - t1 ** 2 * h(l, t1, 0) - t2 ** 2 * h(l, 0, t2)
                        + l ** 4 * (norm.pdf(0, t2, l) + norm.pdf(t1, 0, l) - norm.pdf(t1, t2, l)))

        s3 = 0.5 * l ** 2 * (h(l, t1, t2) - h(l, 0, t2) - h(l, t1, 0))

        s4 = (l / np.sqrt(2 * np.pi)) * (t1 * t2 - l ** 2 / 3)

        return np.sqrt(2 * np.pi) * l * var * (s1 + s2 + s3 - s4)

    @staticmethod
    @lru_cache
    def _h2(l, t):
        """Helper function for _scalar_kernel."""        
        return t * IntegratedGPSE._h(l, t, 0) + l**2 * norm.cdf(t / l) - l**2 * norm.cdf(0)


class DynamicsInformedTwiceIntegratedGPSE(DynamicsInformedTwiceIntegratedGP):
    @staticmethod
    @lru_cache
    def _scalar_kernel(t1, t2, dynamics_coeff, gp_coeff, length_scale, kernel_variance):
        a = dynamics_coeff
        b = gp_coeff
        l = length_scale
        var = kernel_variance
        gma = -l * a / np.sqrt(2)
        return -((np.sqrt(2 * np.pi) * (b**2) * var * l * np.exp(gma**2))/(4 * a))\
                * (DynamicsInformedTwiceIntegratedGPSE._h(l, a, t1, t2)
                   + DynamicsInformedTwiceIntegratedGPSE._h(l, a, t2, t1))

    @staticmethod
    @lru_cache
    def _h(l, a, t1, t2):
        gma = -l * a / np.sqrt(2)
        l_s = l * np.sqrt(2)
        diff_s = (t2 - t1) / l_s
        t1_s = t1 / l_s
        t2_s = t2 / l_s

        s1 = (
            diff_s * erf(diff_s) 
            + t1_s * erf(-t1_s) 
            - t2_s * erf(t2_s) 
            + l_s * (norm.pdf(t2, t1, l) - norm.pdf(t1, 0, l) - norm.pdf(t2, 0, l))
            + 1 / np.sqrt(np.pi)
        )

        s2 = (- np.exp(a*(t2 - t1)) * (erf(diff_s - gma) + erf(t1_s + gma))
            + (2*np.exp(a * t2) - np.exp(a * (t1 + t2)) ) * (erf(t2_s - gma) + erf(gma))
            + np.exp(-gma**2) * ((np.exp(a * t1) - 2) * erf(t2_s)
                                + np.exp(a * t2) * erf(t1_s)
                                + erf(diff_s)                              
                                )
            )

        result = (l_s * np.exp(-gma**2) / a) * s1  + (1 / (a**2)) * (s2)
        return result


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
