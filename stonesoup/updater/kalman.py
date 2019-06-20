# -*- coding: utf-8 -*-

import numpy as np
from functools import lru_cache

from ..base import Property
from .base import Updater
from ..types.prediction import GaussianMeasurementPrediction
from ..types.update import GaussianStateUpdate
from ..models.measurement.linear import LinearGaussian
from ..models.measurement import MeasurementModel
from ..functions import gauss2sigma, unscented_transform


class KalmanUpdater(Updater):
    r"""
    An class which embodies Kalman-type updaters; also a class which performs
    measurement update step as in the standard Kalman Filter. The observation
    model assumes

    .. math::

        \mathbf{z} = h(\mathbf{x}) + \sigma

    with the specific case of the Kalman updater having :math:`h(\mathbf{x}) =
    H \mathbf{x}` and :math:`\sigma = \mathcal{N}(0,R)`. Daughter classes can
    overwrite to specify the measurement model :math:`h(\mathbf{x})`.

    The :attr:`update()` function first calls the :attr:`predict_measurement()`
     function which proceeds by calculating the predicted measurement,
     innovation covariance and measurement cross-covariance,

    .. math::

        \mathbf{z}_{k|k-1} = H_k \mathbf{x}_{k|k-1}

        S_k = H_k P_{k|k-1} H_k^T + R_k

        \Upsilon_k = P_{k|k-1} H_k^T

    where :math:`P_{k|k-1}` is the predicted state covariance. The
    :attr:`predict_measurement()` function returns a
    :class:`GaussianMeasurementPrediction`. The Kalman gain is then calculated
    as,

    .. math::

        K_k = \Upsilon_k S_k^{-1}

    and the posterior state mean and covariance are,

    .. math::

        \mathbf{x}_{k|k} = \mathbf{x}_{k|k-1} + K_k (\mathbf{z}_k - H_k
        \mathbf{x}_{k|k-1})

        P_{k|k} = P_{k|k-1} - K_k S_k K_k^T

    These are returned as a :class:`GaussianStateUpdate` object.
    """

    # TODO: at present this will throw an error if a measurement model is not
    # TODO: specified in either individual measurements or the Updater object
    measurement_model = Property(LinearGaussian, default=None,
                                 doc="A linear Gaussian measurement model. "
                                     "This need not be defined if a "
                                     "measurement model is provided in the "
                                     "measurement. If no model specified on "
                                     "construction, or in the measurement, "
                                     "then error will be thrown.")

    def _check_measurement_model(self, measurement_model):
        """
        Check that the measurement model passed actually exists. If not attach
        the one in the updater. If that one's not specified, return an error.

        :param measurement_model: a measurement model to be checked
        :return: the measurement model to be used
        """
        if measurement_model is None:
            if self.measurement_model is None:
                raise ValueError("No measurement model specified")
            else:
                measurement_model = self.measurement_model

        return measurement_model

    def measurement_matrix(self, predicted_state=None, measurement_model=None,
                           **kwargs):
        r"""
        This is straightforward Kalman so just get the Matrix from the
        measurement model

        :param predicted_state: The predicted state :math:`\mathbf{x}_{k|k-1}`
        :param measurement_model: The measurement model. If omitted the model
        in the updater object is used
        :param kwargs: passed to :class:`~.MeasurementModel`. :attr:`matrix()`

        :return: the measurement matrix, :math:`H_k`
        """
        return self.measurement_model.matrix(**kwargs)

    @lru_cache()
    def predict_measurement(self, predicted_state, measurement_model=None,
                            **kwargs):
        r"""
        Predict the mean measurement implied by the predicted state

        :param predicted_state: The predicted state :math:`\mathbf{x}_{k|k-1}`
        :param measurement_model: The measurement model. If omitted the model
        in the updater object is used
        :param kwargs: passed to :class:`~.MeasurementModel`.
        :attr:`function()` and :class:`~.MeasurementModel`. :attr:`matrix()`

        :return: The :class:`~.GaussianMeasurementPrediction`,
        :math:`\mathbf{z}_{k|k-1}`
        """
        # If a measurement model is not specified then use the one that's
        # native to the updater
        measurement_model = self._check_measurement_model(measurement_model)

        pred_meas = measurement_model.function(predicted_state.state_vector,
                                               noise=0, **kwargs)

        hh = self.measurement_matrix(predicted_state=predicted_state,
                                     measurement_model=measurement_model,
                                     **kwargs)

        innov_cov = hh @ predicted_state.covar @ hh.T + \
            measurement_model.covar()
        meas_cross_cov = predicted_state.covar @ hh.T

        return GaussianMeasurementPrediction(pred_meas, innov_cov,
                                             predicted_state.timestamp,
                                             cross_covar=meas_cross_cov)

    def update(self, hypothesis, force_symmetric_covariance=False, **kwargs):
        r"""
        The Kalman update method

        :param hypothesis: the predicted measurement-measurement association
        hypothesis
        :param symmetric_covariance: force the output covariance
        matrix to be symmetric
        :param kwargs: passed to :attr:`predict_measurement()` of
        :class:`~.KalmanUpdater` or its daughter classes

        :return: Posterior :class:`~.GaussianStateUpdate`,
        :math:`\mathbf{x}_{k|k}`
        """

        # Get the predicted state out of the hypothesis
        predicted_state = hypothesis.prediction

        # If there is no measurement prediction in the hypothesis then do the
        # measurement prediction (and attach it back to the hypothesis).
        if hypothesis.measurement_prediction is None:
            # Get the measurement model out of the measurement if it's there.
            # If not, use the one native to the updater (which might still be
            # none)
            measurement_model = hypothesis.measurement.measurement_model
            measurement_model = self._check_measurement_model(
                measurement_model)

            # Attach the measurement prediction to the hypothesis
            hypothesis.measurement_prediction = self.predict_measurement(
                predicted_state, measurement_model=measurement_model, **kwargs)

        # Get the predicted measurement mean, innovation covariance and
        # measurement cross-covariance
        pred_meas = hypothesis.measurement_prediction.state_vector
        innov_cov = hypothesis.measurement_prediction.covar
        m_cross_cov = hypothesis.measurement_prediction.cross_covar

        # Complete the calculation of the posterior
        kalman_gain = m_cross_cov @ np.linalg.inv(innov_cov)  # This isn't
        # optimised
        posterior_mean = predicted_state.state_vector + \
            kalman_gain @ (hypothesis.measurement.state_vector - pred_meas)
        posterior_covariance = \
            predicted_state.covar - kalman_gain @ innov_cov @ kalman_gain.T

        if force_symmetric_covariance:
            posterior_covariance = (posterior_covariance +
                                    posterior_covariance.T) / 2

        return GaussianStateUpdate(posterior_mean, posterior_covariance,
                                   hypothesis,
                                   hypothesis.measurement.timestamp)


class ExtendedKalmanUpdater(KalmanUpdater):
    r"""
    The EKF version of the Kalman Updater. See description in
    :class:`~.KalmanUpdater`.

    The measurement model may be non-linear but must be differentiable and
    return the linearisation of :math:`h(\mathbf{x})` via the matrix :math:`H`
    accessible via the :attr:`jacobian()` function.

    """
    # TODO: Enforce the fact that this version of MeasurementModel must be
    # TODO: capable of executing :attr:`jacobian()`
    measurement_model = Property(MeasurementModel, default=None,
                                 doc="A measurement model. This need not be "
                                     "defined if a measurement model is "
                                     "provided in the measurement. If no "
                                     "model specified on construction, or in "
                                     "the measurement, then error will be "
                                     "thrown. Must possess :attr:`jacobian()` "
                                     "function.")

    def measurement_matrix(self, predicted_state, measurement_model=None,
                           **kwargs):
        r"""
        Return the (approximate via :attr:`jacobian()`) measurement matrix

        :param predicted_state: the predicted state, :math:`\mathbf{x}_{k|k-1}`
        :param measurement_model: the measurement model. If :attr:`None`
        defaults to the model defined in updater
        :param kwargs: passed to :class:`~.MeasurementModel`. :attr:`matrix()`
         if linear or :class:`~.MeasurementModel`. :attr:`jacobian()` if not

        :return: the measurement matrix, :math:`H_k`
        """

        measurement_model = self._check_measurement_model(measurement_model)

        if hasattr(measurement_model, 'matrix'):
            return measurement_model.matrix(**kwargs)
        else:
            return measurement_model.jacobian(predicted_state.state_vector,
                                              **kwargs)


class UnscentedKalmanUpdater(KalmanUpdater):
    """Unscented Kalman Updater. See description in :class:`~.KalmanUpdater`.

    Perform measurement update step in the Unscented Kalman Filter. The
    :attr:`predict_measurement()` function uses the unscented transform to
    estimate a Gauss-distributed predicted measurement. This is then updated
    via the standard Kalman update equations.
    """
    # Can be linear and non-differentiable
    measurement_model = Property(MeasurementModel, default=None,
                                 doc="The measurement model to be used. This "
                                     "need not be defined if a measurement "
                                     "model is provided in the measurement. "
                                     "If no model specified on construction, "
                                     "or in the measurement, then error will "
                                     "be thrown.")

    alpha = Property(float, default=0.5,
                     doc="Primary sigma point spread scaling parameter, "
                         "typically :math:`10^{-3}`")
    beta = Property(float, default=2,
                    doc="Used to incorporate prior knowledge of the "
                        "distribution. If the true distribution is Gaussian, "
                        "the value of  is optimal.")
    kappa = Property(float, default=0,
                     doc="Secondary spread scaling parameter\
                        (default is calculated as :math:`3-Ns`)")

    # This to ensure that no noise is added to the measurement in the unscented
    # transform (below).
    # Would resolve if the default was to add no noise...
    def measurement_function_nonoise(self, x, w=0, **kwargs):

        return self.measurement_model.function(x, w, **kwargs)

    @lru_cache()
    def predict_measurement(self, predicted_state, measurement_model=None):
        """Unscented Kalman Filter measurement prediction step. Uses the
        unscented transform to estimate a Gauss-distributed predicted
        measurement.

        :param predicted_state: :class:`~.GaussianStatePrediction`, a predicted
         state object
        :param measurement_model: :class:`~.MeasurementModel`, the measurement
        model used to generate the measurement prediction. Should be used in
         cases where the measurement model is dependent on the received
         measurement (the default is ``None``, in which case the updater will
         use the measurement model specified on initialisation)

        :return: :class:`~.GaussianMeasurementPrediction`, the measurement
        prediction

        """
        # If a measurement model is not specified then use the one that's
        # native to the updater
        measurement_model = self._check_measurement_model(measurement_model)

        sigma_points, mean_weights, covar_weights = \
            gauss2sigma(predicted_state.state_vector, predicted_state.covar,
                        self.alpha, self.beta, self.kappa)

        meas_pred_mean, meas_pred_covar, cross_covar, _, _, _ = \
            unscented_transform(sigma_points, mean_weights, covar_weights,
                                self.measurement_function_nonoise,
                                covar_noise=measurement_model.covar())

        return GaussianMeasurementPrediction(meas_pred_mean, meas_pred_covar,
                                             predicted_state.timestamp,
                                             cross_covar)
