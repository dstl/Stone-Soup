# -*- coding: utf-8 -*-

import scipy as sp
from scipy.stats import multivariate_normal

from ...base import Property
from ...types import CovarianceMatrix
from ..base import LinearModel, GaussianModel, TimeVariantModel
from .base import TransitionModel


class LinearTransitionModel(TransitionModel):
    """Linear Transition Model

    Transitions state using a transition matrix, and optionally accepts a
    control matrix and vector to apply to the new state.

    .. math::

        \hat{\mathbf{x}}_{k\mid k-1} &=
            \mathbf{F}_k \hat{\mathbf{x}}_{k-1\mid k-1} +
            \mathbf{B}_k \mathbf{u}_k

    """

    transition_matrix = Property(
        sp.ndarray, doc="Linear transition matrix :math:`\mathbf{F}_k`.")
    control_matrix = Property(
        sp.ndarray, default=None, doc="Control matrix :math:`\mathbf{B}_k`.")

    def transition(self, state_vector, control_vector=None):
        """Transition state

        Parameters
        ----------
        state_vector : StateVector
            State vector :math:`\hat{\mathbf{x}}_{k-1|k-1}`.
        control_vector : StateVector, optional
            Control vector :math:`\mathbf{u}_k`. Default is None in which case
            no control vector is applied.

        Returns
        -------
        StateVector
            New state vector :math:`\hat{\mathbf{x}}_{k\mid k-1}`.
        """
        new_state_vector = self.transition_matrix @ state_vector
        if control_vector is not None:
            new_state_vector += self.control_matrix @ control_vector
        return new_state_vector


class ConstantVelocity1D(TransitionModel, LinearModel,
                         GaussianModel, TimeVariantModel):
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
                dx_{vel} & = & q*dW_t,\ W_t \sim N(0,q^2) & | Speed \ on \
                X-axis (m/s)
            \end{eqnarray}

    Or equivalently:

        .. math::
            x_t = F_t*x_{t-1} + w_t,\ w_t \sim N(0,Q_t)

    where:

        .. math::
            x & = & \begin{bmatrix}
                        x_{pos} \\
                        x_{vel}
                \end{bmatrix}

        .. math::
            F_t & = & \begin{bmatrix}
                        1 & t\\
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
            The number of model state dimensions
        """

        return 2

    def matrix(self, time_interval, **kwargs):
        """Model matrix :math:`F(t)`

        Parameters
        ----------
        time_interval: :class:`datetime.timedelta`
            The time interval

        Returns
        -------
        :class:`numpy.ndarray` of shape (2,2)
            The model matrix evaluated given the provided time interval.
        """

        return sp.array([[1, time_interval.total_seconds()], [0, 1]])

    def function(self, state_vector, time_interval,
                 noise=None, **kwargs):
        """Model function :math:`f(t,x(t),w(t))`

        Parameters
        ----------
        state_vector: class:`stonesoup.types.state.StateVector`
            An input state vector
        time_interval: :class:`datetime.timedelta`
            The time interval
        noise: :class:`numpy.ndarray`
            An externally generated random process noise sample (the default in
            `None`, in which case process noise will be generated internally)

        Returns
        -------
        :class:`numpy.ndarray` of shape (2,2)
            The model fuction evaluated given the provided time interval.
        """

        if noise is None:
            noise = self.rvs(time_interval=time_interval)

        return self.matrix(time_interval,
                           **kwargs)@state_vector + noise

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

    def rvs(self, time_interval, num_samples=1, **kwargs):
        """ Model noise/sample generation function

        Generates noisy samples from the transition model.

        In mathematical terms, this can be written as:

        .. math::

            w_t \sim \mathcal{N}(0,Q_t)

        where :math:`w_t =` ``noise``.

        Parameters
        ----------
        time_interval : :class:`datetime.timedelta`
            A time variant :math:`t`
        num_samples: :class:`int`, optional
            The number of samples to be generated (the default is 1)

        Returns
        -------
        noise : :class:`numpy.ndarray` of shape\
        (:py:attr:`~ndim_state`, ``num_samples``)
            A set of Np samples, generated from the model's noise distribution.
        """

        noise = sp.array([multivariate_normal.rvs(
            sp.array([0, 0]),
            self.covar(time_interval),
            num_samples)]).T

        return noise

    def pdf(self, state_vector_post, state_vector_prior,
            time_interval, **kwargs):
        """ Model pdf/likelihood evaluation function

        Evaluates the pdf/likelihood of the transformed state ``state_post``,
        given the prior state ``state_prior``.

        In mathematical terms, this can be written as:

        .. math::

            p = p(x_t | x_{t-1}) = \mathcal{N}(x_t; x_{t-1}, Q_t)

        where :math:`x_t` = ``state_post``, :math:`x_{t-1}` = ``state_prior``
        and :math:`Q_t` = :py:attr:`~covar`.

        Parameters
        ----------
        state_vector_post : :class:`stonesoup.types.state.State`
            A predicted/posterior state
        state_vector_prior : :class:`stonesoup.types.state.State`
            A prior state
        time_interval: :class:`datetime.timedelta`

        Returns
        -------
        :class:`float`
            The likelihood of ``state_vec_post``, given ``state_vec_prior``
        """

        likelihood = multivariate_normal.pdf(
            state_vector_post.T,
            mean=self.function(state_vector_prior,
                               time_interval,
                               noise=0).ravel(),
            cov=self.covar(time_interval)
        ).T
        return likelihood
