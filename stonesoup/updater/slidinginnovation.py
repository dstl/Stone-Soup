import numpy as np

from ..base import Property
from ..types.array import CovarianceMatrix
from .kalman import KalmanUpdater, ExtendedKalmanUpdater


class SlidingInnovationUpdater(KalmanUpdater):
    r"""Sliding Innovation Filter Updater

    The Sliding Innovation Filter (SIF) is a sub-optimal filter (in comparison to Kalman filter)
    which uses a switching gain to provide robustness to estimation problems that may be
    ill-conditioned or contain modeling uncertainties or disturbances.

    The main difference from Kalman filter is the calculation of the gain:

    .. math::

        K_k = H_k^+ \overline{sat}(|\mathbf{z}_{k|k-1}|/\mathbf{\delta})

    where :math:`\mathbf{\delta}` is the sliding boundary layer width.

    References
    ----------
    1. S. A. Gadsden and M. Al-Shabi, "The Sliding Innovation Filter," in IEEE Access, vol. 8,
       pp. 96129-96138, 2020, doi: 10.1109/ACCESS.2020.2995345.
    """
    layer_width: np.ndarray = Property(
        doc="Sliding boundary layer width :math:`\\mathbf{\\delta}`. A tunable parameter in "
            "measurement space. An example initial value provided in original paper is "
            ":math:`10 \\times \\text{diag}(R)`")

    def _posterior_covariance(self, hypothesis):
        measurement_model = self._check_measurement_model(hypothesis.measurement.measurement_model)
        measurement_matrix = self._measurement_matrix(hypothesis.prediction, measurement_model)

        innovation_vector = hypothesis.measurement.state_vector \
            - hypothesis.measurement_prediction.state_vector
        gain = np.linalg.pinv(measurement_matrix) \
            @ np.diag(np.clip(np.abs(innovation_vector)/self.layer_width, -1, 1).ravel())

        I_KH = np.identity(hypothesis.prediction.ndim) - gain@measurement_matrix
        posterior_covariance = \
            I_KH@hypothesis.prediction.covar@I_KH.T + gain@measurement_model.covar()@gain.T

        return posterior_covariance.view(CovarianceMatrix), gain


class ExtendedSlidingInnovationUpdater(SlidingInnovationUpdater, ExtendedKalmanUpdater):
    """Extended Sliding Innovation Filter Updater

    This is the Extended version of the :class:`~.SlidingInnovationUpdater` for non-linear
    measurement models.

    References
    ----------
    1. S. A. Gadsden and M. Al-Shabi, "The Sliding Innovation Filter," in IEEE Access, vol. 8,
       pp. 96129-96138, 2020, doi: 10.1109/ACCESS.2020.2995345.
    """
    pass
