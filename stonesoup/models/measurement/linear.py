import numpy as np

from typing import Sequence
from ...base import Property
from ...types.array import CovarianceMatrix, Matrix
from ..base import LinearModel, GaussianModel
from .base import MeasurementModel
from ...types.state import StateVector


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

    noise_covar: CovarianceMatrix = Property(doc="Noise covariance")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.noise_covar, CovarianceMatrix):
            self.noise_covar = CovarianceMatrix(self.noise_covar)

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

    def function(self, state, noise=False, **kwargs):
        """Model function :math:`h(t,x(t),w(t))`

        Parameters
        ----------
        state: :class:`~.State`
            An input state
        noise: :class:`numpy.ndarray` or bool
            An externally generated random process noise sample (the default is
            `False`, in which case no noise will be added
            if 'True', the output of :meth:`~.Model.rvs` is added)

        Returns
        -------
        :class:`numpy.ndarray` of shape (:py:attr:`~ndim_meas`, 1)
            The model function evaluated given the provided time interval.
        """

        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs(num_samples=state.state_vector.shape[1], **kwargs)
            else:
                noise = 0

        return self.matrix(**kwargs)@state.state_vector + noise

    def covar(self, **kwargs):
        """Returns the measurement model noise covariance matrix.

        Returns
        -------
        :class:`~.CovarianceMatrix` of shape\
        (:py:attr:`~ndim_meas`, :py:attr:`~ndim_meas`)
            The measurement noise covariance.
        """

        return self.noise_covar

class GeneralLinearGaussian(LinearGaussian):
    """
    An extended implementation of the linear-Gaussian measurement model described by
    .. math::

      y_k = H*x_k + b + v_k,\ \ \ \   v(k)\sim \mathcal{N}(0,R),

    where :math:`H` is a measurement matrix, :math:`b` is a bias vector and :math:`v_k` is Gaussian distributed.
    This class permits specification of :math:`H` in two ways: either constructing it internally, using
    :attr:`~.MeasurementModel.mapping` as in :class:`~.LinearGaussian`, or by explicitly specifying the matrix through
    :attr:`~.GeneralLinearGaussian.meas_matrix`. When both attributes are provided, the preference is given to
    :attr:`~.GeneralLinearGaussian.meas_matrix`. Furthermore, unlike :class:`~.LinearGaussian` this implementation
    permits a certain bias :attr:`~.GeneralLinearGaussian.bias_value` in a measurement model.
    """

    mapping: Sequence[int] = Property(default=None, doc="Mapping between measurement and state dimensions")
    meas_matrix: Matrix = Property(default=None, doc="Arbitrary measurement matrix")
    bias_value: StateVector = Property(default=None, doc="Bias value")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.mapping is None and self.meas_matrix is None:
            raise ValueError("self.mapping and self.meas_matrix cannot both be None.")

        if self.meas_matrix is not None:
            if not isinstance(self.meas_matrix, Matrix):
                self.meas_matrix = Matrix(self.meas_matrix)

        if self.meas_matrix is not None \
                and self.meas_matrix.shape[1] != self.ndim_state:
            raise ValueError("meas_matrix should have the same number of "
                             "columns as the number of rows in state_vector")

        if self.bias_value is None:
            self.bias_value = StateVector([0])

        if not isinstance(self.bias_value, StateVector):
            self.bias_value = StateVector(self.bias_value)

    @property
    def ndim_meas(self):
        """ndim_meas getter method

        Returns
        -------
        :class:`int`
            The number of measurement dimensions.
        """
        if self.meas_matrix is None:
            return super().ndim_meas  # implemented via len(self.mapping) as in LinearGaussian class

        return self.meas_matrix.shape[0]

    def matrix(self, **kwargs):
        """Model matrix :math:`H`

        Returns
        -------
        :class:`numpy.ndarray` of shape \
        (:py:attr:`~ndim_meas`, :py:attr:`~ndim_state`)
            The model matrix.
        """

        if self.meas_matrix is None:
            return super().matrix(**kwargs)  # matrix constructed as in LinearGaussian

        return self.meas_matrix

    def bias(self, **kwargs):
        """Model bias :math:`b`

        Returns
        -------
        :class:`numpy.ndarray` of shape \
        (:py:attr:`~ndim_meas`, 1)
            The bias value.
        """
        return self.bias_value

    def function(self, state, noise=False, **kwargs):
        """Model function :math:`H*x_t+b+v_k`

        Parameters
        ----------
        state: :class:`~.State`
            An input state
        noise: :class:`numpy.ndarray` or bool
            An externally generated random process noise sample (the default is
            `False`, in which case no noise will be added
            if 'True', the output of :meth:`~.Model.rvs` is added)

        Returns
        -------
        :class:`numpy.ndarray` of shape (:py:attr:`~ndim_meas`, 1)
            The model function evaluated.
        """

        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs(num_samples=state.state_vector.shape[1], **kwargs)
            else:
                noise = 0

        return self.matrix(**kwargs) @ state.state_vector + self.bias_value + noise