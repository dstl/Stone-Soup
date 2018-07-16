# -*- coding: utf-8 -*-

import scipy as sp
from scipy.stats import multivariate_normal

from ...base import Property
from ...types.array import CovarianceMatrix
from ..base import NonLinearModel, GaussianModel
from .base import MeasurementModel
from ...functions import jacobian as compute_jac


class Polar2CartesianGaussian(MeasurementModel, NonLinearModel, GaussianModel):
    r"""This is a class implementation of a time-invariant 2D Polar-to-Cartesian \
    measurement model, where measurements are assumed to be received in the \
    form of bearing (:math:`\theta`) and range (:math:`r`), with Gaussian \
    noise in each dimension.
    """

    noise_covar = Property(CovarianceMatrix, doc="Noise covariance")

    @property
    def ndim_meas(self):
        """ndim_meas getter method

        Returns
        -------
        :class:`int`
            The number of measurement dimensions
        """

        return 2

    def jacobian(self, state_vector, **kwargs):
        """Model jacobian matrix :math:`H(t)`

        Parameters
        ----------
        state_vector : :class:`~.StateVector`
            An input state vector

        Returns
        -------
        :class:`numpy.ndarray` of shape (1, py:attr:`~ndim_state`)
            The model jacobian matrix evaluated around the given state vector.
        """

        def fun(x):
            return self.function(state_vector, noise=0)

        return compute_jac(fun, state_vector)

    def function(self, state_vector, noise=None, **kwargs):
        """Model function :math:`h(t,x(t),w(t))`

        Parameters
        ----------
        state_vector: :class:`~.StateVector`
            An input state vector
        noise: :class:`numpy.ndarray`
            An externally generated random process noise sample (the default in
            `None`, in which case process noise will be generated internally)

        Returns
        -------
        :class:`numpy.ndarray` of shape (1, :py:attr:`~ndim_state`)
            The model fumction evaluated given the provided time interval.
        """

        if noise is None:
            noise = self.rvs()

        x = state_vector[self.mapping[0]]
        y = state_vector[self.mapping[1]]

        rho = sp.sqrt(x**2 + y**2)
        phi = sp.arctan2(y, x)

        return sp.array([[phi], [rho]]) + noise

    def covar(self, **kwargs):
        """Returns the measurement model noise covariance matrix.

        Returns
        -------
        :class:`~.CovarianceMatrix` of shape\
        (:py:attr:`~ndim_meas`, :py:attr:`~ndim_meas`)
            The measurement noise covariance.
        """

        return self.noise_covar

    def rvs(self, num_samples=1, **kwargs):
        r""" Model noise/sample generation function

        Generates noise samples from the measurement model.

        In mathematical terms, this can be written as:

        .. math::

            v_t \sim \mathcal{N}(0,R_t)

        where :math:`v_t =` ``noise``.

        Parameters
        ----------
        num_samples: scalar, optional
            The number of samples to be generated (the default is 1)

        Returns
        -------
        noise : 2-D array of shape (:py:attr:`~ndim_meas`, ``num_samples``)
            A set of Np samples, generated from the model's noise
            distribution.
        """

        noise = multivariate_normal.rvs(
            sp.zeros(self.ndim_meas), self.covar(), num_samples)

        if num_samples == 1:
            return noise.reshape((-1, 1))
        else:
            return noise.T

    def pdf(self, meas_vec, state_vec, **kwargs):
        r""" Measurement pdf/likelihood evaluation function

        Evaluates the pdf/likelihood of the (set of) measurement vector(s)
        ``meas_vec``, given the (set of) state vector(s) ``state_vec``.

        In mathematical terms, this can be written as:

        .. math::

            p(y_t | x_t) = \mathcal{N}(y_t; x_t, R_t)

        Parameters
        ----------
        meas_vec : :class:`~.StateVector`
            A measurement
        state_vec : :class:`~.StateVector`
            A state

        Returns
        -------
        :class:`float`
            The likelihood of ``meas``, given ``state``
        """

        likelihood = multivariate_normal.pdf(
            meas_vec.T,
            mean=(self.function(state_vec, 0)).ravel(),
            cov=self.covar()
        )
        return likelihood
