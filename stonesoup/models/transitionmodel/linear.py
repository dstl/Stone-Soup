# -*- coding: utf-8 -*-

import scipy as sp
from scipy.stats import multivariate_normal
from scipy.linalg import block_diag

from ...base import Property
from ...types import CovarianceMatrix
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
        """ Model noise/sample generation function

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
        """ Model pdf/likelihood evaluation function

        Evaluates the pdf/likelihood of the transformed state ``state_post``,
        given the prior state ``state_prior``.

        In mathematical terms, this can be written as:

        .. math::

            p = p(x_t | x_{t-1}) = \mathcal{N}(x_t; x_{t-1}, Q)

        where :math:`x_t` = ``state_post``, :math:`x_{t-1}` = ``state_prior``
        and :math:`Q` = :py:attr:`~covar`.

        Parameters
        ----------
        state_vector_post : :class:`stonesoup.types.state.StateVector`
            A predicted/posterior state
        state_vector_prior : :class:`stonesoup.types.state.StateVector`
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
        sp.ndarray, doc="Transition matrix :math:`\mathbf{F}`.")
    control_matrix = Property(
        sp.ndarray, default=None, doc="Control matrix :math:`\mathbf{B}`.")
    covariance_matrix = Property(
        sp.ndarray,
        doc="Transition noise covariance matrix :math:`\mathbf{Q}`.")

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
