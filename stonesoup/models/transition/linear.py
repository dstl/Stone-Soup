# -*- coding: utf-8 -*-

import scipy as sp
from scipy.stats import multivariate_normal
from scipy.linalg import block_diag

from ...base import Property
from ...types.array import CovarianceMatrix
from ..base import (LinearModel, GaussianModel, TimeVariantModel,
                    TimeInvariantModel)
from .base import TransitionModel


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

    def rvs(self, num_samples=1, **kwargs):
        r""" Model noise/sample generation function

        Generates noisy samples from the transition model.

        In mathematical terms, this can be written as:

        .. math::

            w_t \sim \mathcal{N}(0,Q)

        where :math:`w_t =` ``noise``.

        Parameters
        ----------
        num_samples: :class:`int`, optional
            The number of samples to be generated (the default is 1)

        Returns
        -------
        noise : :class:`numpy.ndarray` of shape\
        (:py:attr:`~ndim_state`, ``num_samples``)
            A set of Np samples, generated from the model's noise distribution.
        """

        noise = sp.array([multivariate_normal.rvs(
            sp.zeros(self.ndim_state),
            self.covar(**kwargs),
            num_samples)])

        if num_samples == 1:
                return noise.reshape((-1, 1))
        else:
            return noise.T

    def pdf(self, state_vector_post, state_vector_prior, **kwargs):
        r""" Model pdf/likelihood evaluation function

        Evaluates the pdf/likelihood of the transformed state ``state_post``,
        given the prior state ``state_prior``.

        In mathematical terms, this can be written as:

        .. math::

            p = p(x_t | x_{t-1}) = \mathcal{N}(x_t; x_{t-1}, Q)

        where :math:`x_t` = ``state_post``, :math:`x_{t-1}` = ``state_prior``
        and :math:`Q` = :py:attr:`~covar`.

        Parameters
        ----------
        state_vector_post : :class:`~.StateVector`
            A predicted/posterior state
        state_vector_prior : :class:`~.StateVector`
            A prior state

        Returns
        -------
        : :class:`float`
            The likelihood of ``state_vec_post``, given ``state_vec_prior``
        """

        likelihood = multivariate_normal.pdf(
            state_vector_post.T,
            mean=self.function(state_vector_prior, noise=0, **kwargs).ravel(),
            cov=self.covar(**kwargs)
        )
        return likelihood


class CombinedLinearGaussianTransitionModel(LinearGaussianTransitionModel):
    r"""Combine multiple models into a single model by stacking them.

    The assumption is that all models are Linear and Gaussian.
    Time Variant, and Time Invariant models can be combined together.
    If any of the models are time variant the keyword argument "time_interval"
    must be supplied to all methods
    """

    model_list = Property(
        [LinearGaussianTransitionModel], doc="List of Transition Models.")

    @property
    def ndim_state(self):
        """ndim_state getter method

        Returns
        -------
        : :class:`int`
            The number of combined model state dimensions.
        """
        return sum(model.ndim_state for model in self.model_list)

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

    def covar(self, **kwargs):
        """Returns the transition model noise covariance matrix.

        Parameters
        ----------
        noise_diff_coeff: :class:`float`, optional
            The noise diffusion coefficient (the default is None, in which\
            case the value of :py:attr:`~noise_diff_coeff` will be used)

        Returns
        -------
        : :class:`stonesoup.types.state.CovarianceMatrix` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
            The process noise covariance.
        """

        covar_list = [model.covar(**kwargs) for model in self.model_list]
        return block_diag(*covar_list)


class LinearGaussianTimeInvariantTransitionModel(LinearGaussianTransitionModel,
                                                 TimeInvariantModel):
    r"""Generic Linear Gaussian Time Invariant Transition Model."""

    transition_matrix = Property(
        sp.ndarray, doc="Transition matrix :math:`\\mathbf{F}`.")
    control_matrix = Property(
        sp.ndarray, default=None, doc="Control matrix :math:`\\mathbf{B}`.")
    covariance_matrix = Property(
        sp.ndarray,
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

        Parameters
        ----------
        noise_diff_coeff: :class:`float`, optional
            The noise diffusion coefficient (the default is None, in which\
            case the value of :py:attr:`~noise_diff_coeff` will be used)

        Returns
        -------
        : :class:`stonesoup.types.state.CovarianceMatrix` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
            The process noise covariance.
        """

        return self.covariance_matrix


class ConstantVelocity(LinearGaussianTransitionModel, TimeVariantModel):
    r"""This is a class implementation of a time-variant 1D Linear-Gaussian
    Constant Velocity Transition Model.

    The target is assumed to move with (nearly) constant velocity, where
    target acceleration is model as white noise.

    The model is described by the following SDEs:

        .. math::
            :nowrap:

            \begin{eqnarray}
                dx_{pos} & = & x_{vel}*d & | {Position \ on \
                X-axis (m)} \\
                dx_{vel} & = & q*dW_t,\ W_t \sim \mathcal{N}(0,q^2) & | Speed \
                on\ X-axis (m/s)
            \end{eqnarray}

    Or equivalently:

        .. math::
            x_t = F_t*x_{t-1} + w_t,\ w_t \sim \mathcal{N}(0,Q_t)

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
                \end{bmatrix}*q
    """

    noise_diff_coeff = Property(
        float, doc="The velocity noise diffusion coefficient :math:`q`")

    @property
    def ndim_state(self):
        """ndim_state getter method

        Returns
        -------
        :class:`int`
            :math:`2` -> The number of model state dimensions
        """

        return 2

    def matrix(self, time_interval, **kwargs):
        """Model matrix :math:`F(t)`

        Parameters
        ----------
        time_interval: :class:`datetime.timedelta`
            A time interval :math:`dt`

        Returns
        -------
        :class:`numpy.ndarray` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
            The model matrix evaluated given the provided time interval.
        """

        return sp.array([[1, time_interval.total_seconds()], [0, 1]])

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

        time_interval_sec = time_interval.total_seconds()

        covar = sp.array([[sp.power(time_interval_sec, 3)/3,
                           sp.power(time_interval_sec, 2)/2],
                          [sp.power(time_interval_sec, 2)/2,
                           time_interval_sec]])*self.noise_diff_coeff

        return CovarianceMatrix(covar)


class ConstantAcceleration(LinearGaussianTransitionModel, TimeVariantModel):
    r"""This is a class implementation of a time-variant 1D Constant
    Acceleration Transition Model.

    The target acceleration is modeled as a zero-mean white noise random
    process.

    The model is described by the following SDEs:

        .. math::
            :nowrap:

            \begin{eqnarray}
                dx_{pos} & = & x_{vel}*d & | {Position \ on \
                X-axis (m)} \\
                dx_{vel} & = & x_{acc}*d & | {Speed \
                on\ X-axis (m/s)} \\
                dx_{acc} & = & q*W_t,\ W_t \sim
                \mathcal{N}(0,q^2) & | {Acceleration \ on \ X-axis (m^2/s)}

            \end{eqnarray}

    Or equivalently:

        .. math::
            x_t = F_t*x_{t-1} + w_t,\ w_t \sim \mathcal{N}(0,Q_t)

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
                      \end{bmatrix}*q^2
    """

    noise_diff_coeff = Property(
        float, doc="The acceleration noise diffusion coefficient :math:`q`")

    @property
    def ndim_state(self):
        """ndim_state getter method

        Returns
        -------
        :class:`int`
            :math:`3` -> The number of model state dimensions
        """

        return 3

    def matrix(self, time_interval, **kwargs):
        """Model matrix :math:`F(t)`

        Parameters
        ----------
        time_interval: :class:`datetime.timedelta`
            A time interval :math:`dt`

        Returns
        -------
        :class:`numpy.ndarray` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
            The model matrix evaluated given the provided time interval.
        """

        time_interval_sec = time_interval.total_seconds()

        return sp.array(
                    [[1, time_interval_sec, sp.power(time_interval_sec, 2)],
                     [0, 1, time_interval_sec],
                     [0, 0, 1]])

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

        time_interval_sec = time_interval.total_seconds()

        covar = sp.array(
            [[sp.power(time_interval_sec, 5) / 20,
              sp.power(time_interval_sec, 4) / 8,
              sp.power(time_interval_sec, 3) / 6],
             [sp.power(time_interval_sec, 4) / 8,
              sp.power(time_interval_sec, 3) / 3,
              sp.power(time_interval_sec, 2) / 2],
             [sp.power(time_interval_sec, 3) / 6,
              sp.power(time_interval_sec, 2) / 2,
              time_interval_sec]]) * sp.power(self.noise_diff_coeff, 2)

        return CovarianceMatrix(covar)


class Singer(LinearGaussianTransitionModel, TimeVariantModel):
    r"""This is a class implementation of a time-variant 1D Singer Transition
    Model.

    The target acceleration is modeled as a zero-mean Gauss-Markov random
    process.

    The model is described by the following SDEs:

        .. math::
            :nowrap:

            \begin{eqnarray}
                dx_{pos} & = & x_{vel}*d & | {Position \ on \
                X-axis (m)} \\
                dx_{vel} & = & x_{acc}*d & | {Speed \
                on\ X-axis (m/s)} \\
                dx_{acc} & = & -\alpha*x_{acc}*d + q*W_t,\ W_t \sim
                \mathcal{N}(0,q^2) & | {Acceleration \ on \ X-axis (m^2/s)}

            \end{eqnarray}

    Or equivalently:

        .. math::
            x_t = F_t*x_{t-1} + w_t,\ w_t \sim \mathcal{N}(0,Q_t)

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

        .. math::
            Q_t & = & 2*\alpha*q^2*\begin{bmatrix}
                        \frac{dt^5}{20} & \frac{dt^4}{8} & \frac{dt^3}{6} \\
                        \frac{dt^4}{8} & \frac{dt^3}{3} & \frac{dt^2}{2} \\
                        \frac{dt^3}{6} & \frac{dt^2}{2} & dt
                        \end{bmatrix}
    """

    noise_diff_coeff = Property(
        float, doc="The acceleration noise diffusion coefficient :math:`q`")
    recip_decorr_time = Property(
        float, doc=r"The reciprocal of the decorrelation time :math:`\alpha`")

    @property
    def ndim_state(self):
        """ndim_state getter method

        Returns
        -------
        :class:`int`
            :math:`3` -> The number of model state dimensions
        """

        return 3

    def matrix(self, time_interval, **kwargs):
        """Model matrix :math:`F(t)`

        Parameters
        ----------
        time_interval: :class:`datetime.timedelta`
            A time interval :math:`dt`

        Returns
        -------
        :class:`numpy.ndarray` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
            The model matrix evaluated given the provided time interval.
        """

        time_interval_sec = time_interval.total_seconds()
        recip_decorr_time = self.recip_decorr_time
        recip_decorr_timedt = recip_decorr_time * time_interval_sec

        return sp.array(
            [[1,
              time_interval_sec,
              (recip_decorr_timedt - 1 + sp.exp(-recip_decorr_timedt)) /
              sp.power(recip_decorr_time, 2)],
             [0,
              1,
              (1 - sp.exp(-recip_decorr_timedt)) / recip_decorr_time],
             [0,
              0,
              sp.exp(-recip_decorr_timedt)]])

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

        time_interval_sec = time_interval.total_seconds()
        recip_decorr_time = self.recip_decorr_time
        noise_diff_coeff = self.noise_diff_coeff
        constant_multiplier = 2 * recip_decorr_time * \
            sp.power(noise_diff_coeff, 2)

        covar = sp.array(
            [[sp.power(time_interval_sec, 5) / 20,
              sp.power(time_interval_sec, 4) / 8,
              sp.power(time_interval_sec, 3) / 6],
             [sp.power(time_interval_sec, 4) / 8,
              sp.power(time_interval_sec, 3) / 3,
              sp.power(time_interval_sec, 2) / 2],
             [sp.power(time_interval_sec, 3) / 6,
              sp.power(time_interval_sec, 2) / 2,
              time_interval_sec]]) * constant_multiplier

        return CovarianceMatrix(covar)


class ConstantTurn(LinearGaussianTransitionModel, TimeVariantModel):
    r"""This is a class implementation of a time-variant 2D Constant Turn
    Model.

    The target is assumed to move with (nearly) constant velocity and also
    known (nearly) constant turn rate.

    The model is described by the following SDEs:

        .. math::
            :nowrap:

            \begin{eqnarray}
                dx_{pos} & = & x_{vel}*d & | {Position \ on \
                X-axis (m)} \\
                dx_{vel} & = &-\omega y_{pos}*d & | {Speed \
                on\ X-axis (m/s)} \\
                dy_{pos} & = & y_{vel}*d & | {Position \ on \
                Y-axis (m)} \\
                dy_{vel} & = & \omega x_{pos}*d & | {Speed \
                on\ Y-axis (m/s)}
            \end{eqnarray}

    Or equivalently:

        .. math::
            x_t = F_t*x_{t-1} + w_t,\ w_t \sim \mathcal{N}(0,Q_t)

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
                          q_x^2* \frac{dt^3}{3} & q_x^2* \frac{dt^2}{2} &
                          0 & 0 \\
                          q_x^2 \frac{dt^2}{2} & q_x^2  dt &
                          0 & 0 \\
                          0 & 0 &
                          q_y^2* \frac{dt^3}{3} & q_y^2* \frac{dt^2}{2}\\
                          0 & 0 &
                          q_y^2 \frac{dt^2}{2} & q_y^2 dt
                      \end{bmatrix}
    """

    noise_diff_coeffs = Property(
        sp.ndarray,
        doc="The acceleration noise diffusion coefficients :math:`q`")
    turn_rate = Property(
        float, doc=r"The turn rate :math:`\omega`")

    @property
    def ndim_state(self):
        """ndim_state getter method

        Returns
        -------
        :class:`int`
            :math:`4` -> The number of model state dimensions
        """

        return 4

    def matrix(self, time_interval, **kwargs):
        """Model matrix :math:`F(t)`

        Parameters
        ----------
        time_interval: :class:`datetime.timedelta`
            A time interval :math:`dt`

        Returns
        -------
        :class:`numpy.ndarray` of shape\
        (:py:attr:`~ndim_state`, :py:attr:`~ndim_state`)
            The model matrix evaluated given the provided time interval.
        """

        time_interval_sec = time_interval.total_seconds()
        turn_ratedt = self.turn_rate * time_interval_sec

        return sp.array(
            [[1, sp.sin(turn_ratedt) / self.turn_rate,
              0, -(1 - sp.cos(turn_ratedt)) / self.turn_rate],
             [0, sp.cos(turn_ratedt),
              0, -sp.sin(turn_ratedt)],
             [0, (1 - sp.cos(turn_ratedt)) / self.turn_rate,
              1, sp.sin(turn_ratedt) / self.turn_rate],
             [0, sp.sin(turn_ratedt),
              0, sp.cos(turn_ratedt)]])

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

        time_interval_sec = time_interval.total_seconds()
        base_covar = sp.array([[sp.power(time_interval_sec, 3) / 3,
                                sp.power(time_interval_sec, 2) / 2],
                               [sp.power(time_interval_sec, 2) / 2,
                                time_interval_sec]])
        covar_list = [base_covar*sp.power(self.noise_diff_coeffs[0], 2),
                      base_covar*sp.power(self.noise_diff_coeffs[1], 2)]
        covar = sp.linalg.block_diag(*covar_list)

        return CovarianceMatrix(covar)
