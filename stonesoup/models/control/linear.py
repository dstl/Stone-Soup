# -*- coding: utf-8 -*-

import scipy as sp

from .base import ControlModel
from ..base import LinearModel
from ...base import Property


class LinearControlModel(ControlModel, LinearModel):
    """
    Linear Control Model

    Implements a linear effect to the state vector through the contribution :math:`\hat{x}_k = B_k \mathbf{u}_k`

    where :math:`B_k` is the control-input model matrix (control matrix for short) and :math:`\mathbf{u}_k` is the
    control vector

    """

    control_vector = Property(sp.ndarray, doc="Control vector at time :math:`k`, :math:`\mathbf{u}_k`")
    control_matrix = Property(sp.ndarray, doc="Control-input model matrix at time :math:`k`, :math:`B_k`")
    control_noise = Property(sp.ndarray, doc="Control-input noise covariance at time :math:`k`")

    @property
    def ndim_ctrl(self):
        return self.control_vector.shape[0]

    def matrix(self):
        return self.control_matrix

    # Probably should be defined as a method at a more abstract level
    def control_input(self):
        return self.control_matrix @ self.control_vector

    def rvs(self):
        # Sample (just once at moment) from the multivariate normal distribution suggested by the mean and covariance
        # control parameters.
        return sp.random.multivariate_normal(self.control_input(), self.control_noise)

    def pdf(self):
        # TODO implement this
        pass
