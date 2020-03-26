# -*- coding: utf-8 -*-

from scipy import ndarray
from scipy.stats import multivariate_normal

from .base import ControlModel
from ..base import LinearModel
from ...base import Property


class LinearControlModel(ControlModel, LinearModel):
    r"""Implements a linear effect to the state vector via,

    .. math::

        \hat{x}_k = B_k \mathbf{u}_k + \gamma_k

    where :math:`B_k` is the control-input model matrix (i.e. control matrix),
    :math:`\mathbf{u}_k` is the control vector and :math:`\gamma_k` is
    sampled from zero-mean white noise distribution
    :math:`\mathcal{N}(0,\Gamma_k)`

    """

    control_vector = Property(
        ndarray, doc="Control vector at time :math:`k`")
    control_matrix = Property(
        ndarray,
        doc="Control input model matrix at time :math:`k`, :math:`B_k`")
    control_noise = Property(
        ndarray,
        default=None,
        doc="Control input noise covariance at time :math:`k`")

    @property
    def ndim(self):
        return self.ndim_ctrl

    @property
    def ndim_ctrl(self):
        return self.control_vector.shape[0]

    def matrix(self):
        """
        Returns
        -------
        : :class:`numpy.ndarray`
            the control-input model matrix, :math:`B_k`
        """
        return self.control_matrix

    def control_input(self):
        r"""The mean control input

        Returns
        -------
        : :class:`numpy.ndarray`
            the noiseless effect of the control input, :math:`B_k \mathbf{u}_k`

        """
        return self.control_matrix @ self.control_vector

    def rvs(self):
        r"""Sample (once) from the multivariate normal distribution determined
        from the mean and covariance control parameters

        Returns
        -------
        : :class:`numpy.ndarray`
            a sample from :math:`\mathcal{N}(B_k \mathbf{u}_k, \Gamma_k)`

        """
        return multivariate_normal.rvs(self.control_input(),
                                       self.control_noise).reshape(-1, 1)

    def pdf(self, control_vec):
        """The value of the probability density function (pdf) at a test point

        Parameters
        ----------
        control_vec : :class:`numpy.ndarray`
            The control vector at the test point

        Returns
        -------
        float
            The value of the pdf at :obj:`control_vec`

        """
        return multivariate_normal.pdf(control_vec,
                                       mean=self.control_input(),
                                       cov=self.control_noise).reshape(-1, 1)
