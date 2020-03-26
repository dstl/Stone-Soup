# -*- coding: utf-8 -*-

import numpy as np

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

        model_matrix = np.zeros((self.ndim_meas, self.ndim_state))
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
