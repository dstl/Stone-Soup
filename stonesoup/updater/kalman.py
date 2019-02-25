# -*- coding: utf-8 -*-

import numpy as np

from ..base import Property
from .base import Updater
from ..types import GaussianMeasurementPrediction, GaussianStateUpdate
from ..models import LinearGaussian
from ..models.measurement import MeasurementModel, NonLinearModel


class AbstractKalmanUpdater(Updater):
    """
    An abstract class which embodies much of the functional infrastructure inherent in Kalman-type updaters;
    will allow daughter classes merely to specify the measurement model h(x)

    Note that this isn't an abstract class as such. It will execute the functions
    """

    # TODO at present this will throw an error if a measurement model is not specified. Either remove default=None
    # TODO or specify some sort of behaviour if default is none.
    measurement_model = Property(MeasurementModel, default=None, doc="The measurement model to be used")

    def measurement_matrix(self, predicted_state=None, **kwargs):
        pass

    def predict_measurement(self, predicted_state, **kwargs):
        """

        :param predicted_state: The predicted state :math:`\hat{\mathbf{x}}_{k|k-1}`
        :param kwargs:
        :return: A Gaussian measurement prediction, :math:`\hat{\mathbf{z}}_{k}`
        """

        pred_meas = self.measurement_model.function(predicted_state.state_vector) # I'd prefer to pass `State`

        hh = self.measurement_matrix(predicted_state)

        innov_cov = hh @ predicted_state.covariance() @ hh.T + self.measurement_model.noise_covar
        meas_cross_cov = predicted_state.covariance() @ hh.T

        return GaussianMeasurementPrediction(pred_meas, innov_cov, predicted_state.timestamp, meas_cross_cov)

    def update(self, predicted_state, hypothesis, **kwargs):
        """
        The Kalman-type update method

        :param predicted_state: the predicted state, :math:`\hat{\mathbf{x}}_{k|k-1}`
        :param hypothesis: the predicted measurement-measurement association hypothesis
        :param kwargs:
        :return: Posterior state, :math:`\mathbf{x}_{k|k}`
        """

        gmp = self.predict_measurement(predicted_state, **kwargs)

        pred_meas = gmp.state_vector
        innov_cov = gmp.covar
        m_cross_cov = gmp.cross_cov

        kalman_gain = m_cross_cov @ np.linalg.inv(innov_cov)
        posterior_mean = predicted_state.state_vector + kalman_gain @ (hypothesis.measurement.state_vector - pred_meas)
        posterior_covariance = predicted_state.covariance() - kalman_gain @ innov_cov @ kalman_gain.T

        #posterior_state_covariance = (P_post + P_post.T) / 2 # !!! kludge

        return GaussianStateUpdate(posterior_mean, posterior_covariance, hypothesis, hypothesis.measurement.timestamp)


class KalmanUpdater(AbstractKalmanUpdater):
    """
    Kalman Updater

    Perform measurement update step as in the standard Kalman Filter. Assumes the measurement matrix function of the
    measurement_model returns a matrix (H).

    """

    measurement_model = Property(LinearGaussian, default=None, doc="A linear Gaussian measurement model")

    def measurement_matrix(self):
        """
        This is straightforward Kalman so just get the Matrix from the measurement model
        :return: the measurement matrix, :math:`H_k`
        """
        return self.measurement_model.matrix()


class ExtendedKalmanUpdater(AbstractKalmanUpdater):
    """
    The EKF version of the Kalman Updater

    The measurement model must be non-linear and return the linearisation of h() via the jacobian matrix H

    """

    measurement_model = Property(NonLinearModel, default=None, doc="A non-linear differentiable measurement model")

    def measurement_matrix(self, predicted_state):
        """

        :param predicted_state: the predicted state, :math:`\hat{\mathbf{x}}_{k|k-1}`
        :return: the measurement matrix, :math:`H_k`
        """
        return self.measurement_model.jacobian(predicted_state.state_vector)
