# -*- coding: utf-8 -*-
from abc import abstractmethod

import numpy as np
from scipy.stats import multivariate_normal

from ..base import Base
from ..functions import jacobian as compute_jac
from ..types.numeric import Probability


class Model(Base):
    """Model type

    Base/Abstract class for all models."""

    @property
    @abstractmethod
    def ndim(self):
        """Number of dimensions of model"""
        pass

    @abstractmethod
    def function(self, state_vector, noise=None):
        """ Model function"""
        pass

    @abstractmethod
    def rvs(self, num_samples=1):
        """Model noise/sample generation method"""
        pass

    @abstractmethod
    def pdf(self, state_vector1, state_vector2):
        """Model pdf/likelihood evaluator method"""
        pass


class LinearModel(Model):
    """LinearModel class

    Base/Abstract class for all linear models"""

    @abstractmethod
    def matrix(self):
        """ Model matrix"""
        pass

    def function(self, state_vector, noise=None, **kwargs):
        """Model linear function :math:`f_k(x(k),w(k)) = F_k(x_k) + w_k`

        Parameters
        ----------
        state_vector: :class:`~.StateVector`
            An input state vector
        noise: :class:`numpy.ndarray`
            An externally generated random process noise sample (the default is
            `None`, in which case process noise will be generated via
            :meth:`~.Model.rvs`)

        Returns
        -------
        : :class:`numpy.ndarray`
            The model function evaluated.
        """

        if noise is None:
            # TODO: doesn't make sense for noise=None to generate noise
            noise = self.rvs(**kwargs)
        else:
            noise = 0

        return self.matrix(**kwargs) @ (state_vector + noise)


class NonLinearModel(Model):
    """NonLinearModel class

    Base/Abstract class for all non-linear models"""

    def jacobian(self, state_vector, **kwargs):
        """Model jacobian matrix :math:`H_{jac}`

        Parameters
        ----------
        state_vector : :class:`~.StateVector`
            An input state vector

        Returns
        -------
        :class:`numpy.ndarray` of shape (:py:attr:`~ndim_meas`, \
        :py:attr:`~ndim_state`)
            The model jacobian matrix evaluated around the given state vector.
        """

        def fun(x):
            return self.function(x, noise=0)

        return compute_jac(fun, state_vector)

    @abstractmethod
    def function(self, state_vector, noise=None, **kwargs):
        """Model function :math:`f(t,x(t),w(t))`

        Parameters
        ----------
        state_vector: :class:`~.StateVector`
            An input state vector
        noise: :class:`numpy.ndarray`
            An externally generated random process noise sample (the default in
            `None`, in which case process noise will be generated internally)

        Returns
        -------
        : :class:`numpy.ndarray`
            The model function evaluated.
        """
        pass


class ReversibleModel(NonLinearModel):
    """Non-linear model containing sufficient co-ordinate
    information such that the linear co-ordinate conversions
    can be calculated from the non-linear counterparts.

    Contains an inverse function which computes the reverse
    of the relevant linear-to-non-linear function"""

    @abstractmethod
    def inverse_function(self, state_vector, **kwargs):
        """Takes in the result of the function and
        computes the inverse function, returning the initial
        input of the function.

        Parameters
        ----------
        state_vector: :class:`~.StateVector`
            Input state vector (non-linear format)

        Returns
        -------
        : :class:`numpy.ndarray`
            The linear co-ordinates
        """
        pass


class TimeVariantModel(Model):
    """TimeVariantModel class

    Base/Abstract class for all time-variant models"""


class TimeInvariantModel(Model):
    """TimeInvariantModel class

    Base/Abstract class for all time-invariant models"""


class GaussianModel(Model):
    """GaussianModel class

    Base/Abstract class for all Gaussian models"""

    def rvs(self, num_samples=1, **kwargs):
        r"""Model noise/sample generation function

        Generates noise samples from the model.

        In mathematical terms, this can be written as:

        .. math::

            v_t \sim \mathcal{N}(0,Q)

        where :math:`v_t =` ``noise`` and :math:`Q` = :attr:`covar`.

        Parameters
        ----------
        num_samples: scalar, optional
            The number of samples to be generated (the default is 1)

        Returns
        -------
        noise : 2-D array of shape (:attr:`~.ndim`, ``num_samples``)
            A set of Np samples, generated from the model's noise
            distribution.
        """

        noise = multivariate_normal.rvs(
            np.zeros(self.ndim), self.covar(**kwargs), num_samples)

        return np.atleast_2d(noise).T

    def pdf(self, state_vector1, state_vector2, **kwargs):
        r"""Model pdf/likelihood evaluation function

        Evaluates the pdf/likelihood of ``state_vector1``, given the state
        ``state_vector2`` which is passed to :meth:`~.function`.

        In mathematical terms, this can be written as:

        .. math::

            p = p(y_t | x_t) = \mathcal{N}(y_t; x_t, Q)

        where :math:`y_t` = ``state_vector1``, :math:`x_t` = ``state_vector2``
        and :math:`Q` = :attr:`covar`.

        Parameters
        ----------
        state_vector1 : :class:`~.StateVector`
        state_vector2 : :class:`~.StateVector`

        Returns
        -------
        : :class:`~.Probability`
            The likelihood of ``state_vector1``, given ``state_vector2``
        """

        likelihood = multivariate_normal.logpdf(
            state_vector1.T,
            mean=self.function(state_vector2, noise=0, **kwargs).ravel(),
            cov=self.covar(**kwargs)
        )
        return Probability(likelihood, log_value=True)

    @abstractmethod
    def covar(self):
        """Model covariance"""
