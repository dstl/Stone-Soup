# -*- coding: utf-8 -*-

import scipy as sp
from scipy.stats import multivariate_normal

from ...base import Property
from ...types import CovarianceMatrix
from ..base import LinearModel, GaussianModel
from .base import MeasurementModel


class LinearGaussian1D(MeasurementModel, LinearModel, GaussianModel):
    """This is a class implementation of a time-invariant 1D
    Linear-Gaussian Measurement Model.

    The model is described by the following equations:

    .. math::

      y_t = H_k*x_t + v_k,\ \ \ \   v(k)\sim \mathcal{N}(0,R)

    where ``H_k`` is a (1, py:attr:`~ndim_state`) matrix and ``v_k`` is\
    Gaussian distributed.

    """

    noise_covar = Property(CovarianceMatrix, doc="Noise covariance")

    def __init__(self, ndim_state, mapping, noise_covar, *args, **kwargs):
        """Constructor method"""

        super().__init__(ndim_state, mapping,
                         noise_covar, *args, **kwargs)

    @property
    def ndim_meas(self):
        """ndim_meas getter method

        Returns
        -------
        :class:`int`
            The number of measurement dimensions
        """

        return 1

    def matrix(self, **kwargs):
        """Model matrix :math:`H(t)`

        Returns
        -------
        :class:`numpy.ndarray` of shape (1, py:attr:`~ndim_state`)
            The model matrix evaluated given the provided time interval.
        """

        model_matrix = sp.zeros((1, self.ndim_state))
        model_matrix[0, self.mapping] = 1

        return model_matrix

    def function(self, state_vector, noise=None, **kwargs):
        """Model function :math:`h(t,x(t),w(t))`

        Parameters
        ----------
        state_vector: class:`stonesoup.types.state.StateVector`
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

        return self.matrix(**kwargs)@state_vector + noise

    def covar(self, **kwargs):
        """Returns the measurement model noise covariance matrix.

        Returns
        -------
        :class:`stonesoup.types.state.CovarianceMatrix` of shape\
        (:py:attr:`~ndim_meas`, :py:attr:`~ndim_meas`)
            The measurement noise covariance.
        """

        return CovarianceMatrix(self.noise_covar)

    def rvs(self, num_samples=1, **kwargs):
        """ Model noise/sample generation function

        Generates noise samples from the transition model.

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
            sp.array([0, 0]), self.covar(), num_samples).T

        return noise

    def pdf(self, meas_vec, state_vec, **kwargs):
        """ Measurement pdf/likelihood evaluation function

        Evaluates the pdf/likelihood of the (set of) measurement vector(s)
        ``meas_vec``, given the (set of) state vector(s) ``state_vec``.

        In mathematical terms, this can be written as:

        .. math::

            p(y_t | x_t) = \mathcal{N}(y_t; x_t, R_t)

        Parameters
        ----------
        meas : :class:`stonesoup.types.state.State`
            A measurement
        state : :class:`stonesoup.types.state.State`
            A state

        Returns
        -------
        :class:`float`
            The likelihood of ``meas``, given ``state``
        """

        likelihood = multivariate_normal.pdf(
            meas_vec.T,
            mean=self.matrix()@state_vec.ravel(),
            cov=self.covar()
        ).T
        return likelihood
