# -*- coding: utf-8 -*-

import scipy as sp
from scipy.stats import multivariate_normal

from ...base import Property
from ...types.array import CovarianceMatrix
from ..base import LinearModel, GaussianModel
from .base import MeasurementModel


# TODO: Probably should call this LinearGaussianMeasurementModel
class LinearGaussian(MeasurementModel, LinearModel, GaussianModel):
    r"""This is a class implementation of a time-invariant 1D
    Linear-Gaussian Measurement Model.

    The model is described by the following equations:

    .. math::

      y_t = H_k*x_t + v_k,\ \ \ \   v(k)\sim \mathcal{N}(0,R)

    where ``H_k`` is a (:py:attr:`~ndim_meas`, :py:attr:`~ndim_state`) \
    matrix and ``v_k`` is Gaussian distributed.

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
        return len(self.mapping)

    def matrix(self, **kwargs):
        """Model matrix :math:`H(t)`

        Returns
        -------
        :class:`numpy.ndarray` of shape \
        (:py:attr:`~ndim_meas`, :py:attr:`~ndim_state`)
            The model matrix evaluated given the provided time interval.
        """

        """if the state has acceleration too, need to account for that. Need 1's where there are state dimensions"""

        model_matrix = sp.zeros((self.ndim_meas, self.ndim_state))
        for dim_meas, dim_state in enumerate(self.mapping):
            if dim_state is not None:
                model_matrix[dim_meas, dim_state] = 1

        return model_matrix

    def function(self, state_vector, noise=None, **kwargs):
        """Model function :math:`h(t,x(t),w(t))`

        Parameters
        ----------
        state_vector: :class:`~.StateVector`
            An input state vector
        noise: :class:`numpy.ndarray`
            An externally generated random process noise sample (the default in
            `None`, in which case process noise will be added via :meth:`rvs`)

        Returns
        -------
        :class:`numpy.ndarray` of shape (:py:attr:`~ndim_meas`, 1)
            The model function evaluated given the provided time interval.
        """

        if noise is None:
            noise = self.rvs()  # TODO: change noise=None generates noise!

        return self.matrix(**kwargs)@state_vector + noise

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

        if len(state_vec) == 6:
            """model_matrix = sp.zeros((self.ndim_meas, self.ndim_state))
            for dim_meas, dim_state in enumerate(self.mapping):
                if dim_state is not None:
                    model_matrix[dim_meas, dim_state] = 1"""

            likelihood = multivariate_normal.pdf(
                meas_vec.T,
                mean=(self.matrix()@state_vec).ravel(),
                cov=self.covar()
            )
        else:
            model_matrix = sp.zeros((self.ndim_meas, len(state_vec)))
            for dim_meas, dim_state in enumerate((0, 3, 6)):
                if dim_state is not None:
                    model_matrix[dim_meas, dim_state] = 1

            likelihood = multivariate_normal.pdf(
                meas_vec.T,
                mean=(model_matrix @ state_vec).ravel(),
                cov=self.covar()
            )

            return likelihood
