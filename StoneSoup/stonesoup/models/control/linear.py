import numpy as np

from .base import ControlModel
from ..base import LinearModel, GaussianModel
from ...base import Property
from ...types.array import StateVector


class LinearControlModel(ControlModel, LinearModel, GaussianModel):
    r"""Implements a linear effect to the state vector via,

    .. math::

        \hat{x}_k = B_k (\mathbf{u}_k + \gamma_k)

    where :math:`B_k` is the control-input model matrix (i.e. control matrix),
    :math:`\mathbf{u}_k` is the control vector and :math:`\gamma_k` is
    sampled from zero-mean white noise distribution
    :math:`\mathcal{N}(0,\Gamma_k)`

    """

    control_matrix: np.ndarray = Property(
        doc="Control input model matrix at time :math:`k`, :math:`B_k`")
    control_noise: np.ndarray = Property(
        default=None,
        doc="Control input noise covariance at time :math:`k`")

    def __init__(self, *args, **kwargs):
        """Ensures that the None control noise defaults to a ndimxndim zero matrix"""
        super().__init__(*args, **kwargs)

        if self.control_noise is None:
            self.control_noise = np.zeros([self.ndim_ctrl, self.ndim_ctrl])

    @property
    def ndim(self):
        return self.ndim_ctrl

    @property
    def ndim_ctrl(self):
        return self.control_matrix.shape[1]

    def matrix(self, **kwargs) -> np.ndarray:
        """
        Returns
        -------
        : :class:`numpy.ndarray`
            the control-input model matrix, :math:`B_k`
        """

        return self.control_matrix

    def covar(self, **kwargs):

        return self.control_noise

    def function(self, control_input, noise=False, **kwargs) -> StateVector:
        """This needs to be overwritten because noise is added before the transformation
        rather than after it.

        """
        # have to accept that control input might be None and then adjust (including to add noise).
        if control_input is None:
            control_vector = StateVector(np.zeros(self.ndim_ctrl))
        else:
            control_vector = control_input.state_vector

        if isinstance(noise, bool) or noise is None:
            if noise:
                noise = self.rvs(num_samples=control_vector.shape[1], **kwargs)
            else:
                noise = 0

        return self.matrix(**kwargs) @ (control_vector + noise)
