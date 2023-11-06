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
