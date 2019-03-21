# -*- coding: utf-8 -*-

import numpy as np

from ..base import Property
from .base import Updater
from ..types import GaussianMeasurementPrediction, GaussianStateUpdate
from ..models import LinearGaussian
from ..models.measurement import MeasurementModel
from ..functions import gauss2sigma, unscented_transform


class AbstractKalmanUpdater(Updater):
    """
    An abstract class which embodies much of the functional infrastructure inherent in Kalman-type updaters;
    will allow daughter classes merely to specify the measurement model h(x)

    Note that this isn't an abstract class as such. It will execute the functions
    """

    # TODO at present this will throw an error if a measurement model is not specified in either individual
    # TODO measurements or the Updater object
    measurement_model = Property(MeasurementModel, default=None, doc="The measurement model to be used")

    def measurement_matrix(self, predicted_state, measurement_model=None, **kwargs):
        pass

    def predict_measurement(self, predicted_state, measurement_model=None, **kwargs):
        """
        :param predicted_state: The predicted state :math:`\hat{\mathbf{x}}_{k|k-1}`
        :param measurement_model: The measurement model. If omitted the model in the updater object is used
        :param kwargs:
        :return: A Gaussian measurement prediction, :math:`\hat{\mathbf{z}}_{k}`
        """
        # If a measurement model is not specified then use the one that's native to the updater
        if measurement_model is None:
            measurement_model = self.measurement_model

        pred_meas = measurement_model.function(predicted_state.state_vector, measurement_model=measurement_model,
                                               noise=[0])

        hh = self.measurement_matrix(predicted_state, measurement_model)

        innov_cov = hh @ predicted_state.covar @ hh.T + measurement_model.covar()
        meas_cross_cov = predicted_state.covar @ hh.T

        return GaussianMeasurementPrediction(pred_meas, innov_cov, predicted_state.timestamp, meas_cross_cov)

    def update(self, hypothesis, **kwargs):
        """
        The Kalman-type update method

        :param hypothesis: the predicted measurement-measurement association hypothesis
        :param kwargs:
        :return: Posterior state, :math:`\mathbf{x}_{k|k}` as :class:`~.GaussianMeasurementPrediction`
        """

        predicted_state = hypothesis.prediction  # Get the predicted state out of the hypothesis

        # Get the measurement model out of the measurement if it's there. If not, use the one native to the updater
        # (which might still be none)
        measurement_model = hypothesis.measurement.measurement_model
        if measurement_model is None:
            measurement_model = self.measurement_model

        # Get the measurement prediction
        gmp = self.predict_measurement(predicted_state, measurement_model=measurement_model, **kwargs)
        pred_meas = gmp.state_vector
        innov_cov = gmp.covar
        m_cross_cov = gmp.cross_covar

        # Complete the calculation of the posterior
        kalman_gain = m_cross_cov @ np.linalg.inv(innov_cov)  # This isn't optimised
        posterior_mean = predicted_state.state_vector + kalman_gain @ (hypothesis.measurement.state_vector - pred_meas)
        posterior_covariance = predicted_state.covar - kalman_gain @ innov_cov @ kalman_gain.T

        # posterior_state_covariance = (P_post + P_post.T) / 2 # !!! kludge

        return GaussianStateUpdate(posterior_mean, posterior_covariance, hypothesis, hypothesis.measurement.timestamp)


class KalmanUpdater(AbstractKalmanUpdater):
    """
    Kalman Updater

    Perform measurement update step as in the standard Kalman Filter. Assumes the measurement matrix function of the
    measurement_model returns a matrix (H).

    """

    measurement_model = Property(LinearGaussian, default=None, doc="A linear Gaussian measurement model")

    def measurement_matrix(self, **kwargs):
        """
        This is straightforward Kalman so just get the Matrix from the measurement model
        :return: the measurement matrix, :math:`H_k`
        """
        return self.measurement_model.matrix()


class ExtendedKalmanUpdater(AbstractKalmanUpdater):
    """
    The EKF version of the Kalman Updater

    The measurement model may be non-linear but must be differentiable and return the linearisation of h() via the
    matrix H accessible via the :attr:`jacobian()` function.

    """

    def measurement_matrix(self, predicted_state, measurement_model=None, **kwargs):
        """

        :param predicted_state: the predicted state, :math:`\hat{\mathbf{x}}_{k|k-1}`
        :param measurement_model: the measurement model. If `None` defaults to the model defined in updater
        :return: the measurement matrix, :math:`H_k`
        """
        if measurement_model is None:
            measurement_model = self.measurement_model

        if hasattr(measurement_model, 'matrix'):
            return measurement_model.matrix()
        else:
            return measurement_model.jacobian(predicted_state.state_vector)


class UnscentedKalmanUpdater(AbstractKalmanUpdater):
    """Unscented Kalman Updater

    Perform measurement update step in the Unscented Kalman Filter. The predict measurement step uses the unscented
    transform to estimate a Gauss-distributed predicted measurement. This is then updated via the standard Kalman
    updater.
    """

    alpha = Property(float, default=0.5,
                     doc="Primary sigma point spread scalling parameter.\
                         Typically 1e-3.")
    beta = Property(float, default=2,
                    doc="Used to incorporate prior knowledge of the distribution.\
                        If the true distribution is Gaussian, the value of 2\
                        is optimal.")
    kappa = Property(float, default=0,
                     doc="Secondary spread scaling parameter\
                        (default is calculated as 3-Ns)")

    def predict_measurement(self, predicted_state, measurement_model=None, **kwargs):

        """Unscented Kalman Filter measurement prediction step

        Parameters
        ----------
        predicted_state : :class:`~.GaussianStatePrediction`
            A predicted state object
        measurement_model: :class:`~.MeasurementModel`, optional
            The measurement model used to generate the measurement prediction.Should be used in cases where the
            measurement model is dependent on the received measurement (the default is ``None``, in which case the
            updater will use the measurement model specified on initialisation)

        Returns
        -------
        : :class:`~.GaussianMeasurementPrediction`
            The measurement prediction
        """
        # If a measurement model is not specified then use the one that's native to the updater
        if measurement_model is None:
            measurement_model = self.measurement_model

        sigma_points, mean_weights, covar_weights = \
            gauss2sigma(predicted_state.state_vector, predicted_state.covar, self.alpha, self.beta, self.kappa)

        meas_pred_mean, meas_pred_covar, cross_covar, _, _, _ = \
            unscented_transform(sigma_points, mean_weights, covar_weights, measurement_model.function,
                                covar_noise=measurement_model.covar())

        return GaussianMeasurementPrediction(meas_pred_mean, meas_pred_covar, predicted_state.timestamp, cross_covar)
