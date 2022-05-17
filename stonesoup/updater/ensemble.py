# -*- coding: utf-8 -*-
import scipy

from .kalman import KalmanUpdater
from ..base import Property
from ..types.state import State
from ..types.array import StateVectors
from ..types.prediction import MeasurementPrediction
from ..types.update import Update
from ..models.measurement import MeasurementModel


class EnsembleUpdater(KalmanUpdater):
    r"""Ensemble Kalman Filter Updater class

    The EnKF is a hybrid of the Kalman updating scheme and the
    Monte Carlo aproach of the the particle filter.

    Deliberately structured to resemble the Vanilla Kalman Filter,
    :meth:`update` first calls :meth:`predict_measurement` function which
    proceeds by calculating the predicted measurement, innovation covariance
    and measurement cross-covariance. Note however, these are not propagated
    explicitly, they are derived from the sample covariance of the ensemble
    itself.

    Note that the EnKF equations are simpler when written in the following
    formalism. Note that h is not neccisarily a matrix, but could be a
    nonlinear measurement function.

    .. math::

        \mathbf{A}_k = \hat{X} - E(X)

        \mathbf{HA}_k = h(\hat{X} - E(X))

    The cross covariance and measurement covariance are given by:

    .. math::

        P_{xz} = \frac{1}{M-1} \mathbf{A}_k \mathbf{HA}_k^T

        P_{zz} = \frac{1}{M-1} A_k \mathbf{HA}_k^T + R

    The Kalman gain is then calculated via:

    .. math::
        K_{k} = P_{xz} P_{zz}^{-1}

    and the posterior state mean and covariance are,

    .. math::

        \mathbf{x}_{k|k} = \mathbf{x}_{k|k-1} + K_k (\mathbf{z}_k - H_k
        \mathbf{x}_{k|k-1})

    This is returned as a :class:`~.EnsembleStateUpdate` object.

    References
    ----------
    1. J. Hiles, S. M. O’Rourke, R. Niu and E. P. Blasch,
    "Implementation of Ensemble Kalman Filters in Stone-Soup,"
    International Conference on Information Fusion, (2021)

    2. Mandel, Jan. "A brief tutorial on the ensemble Kalman filter."
    arXiv preprint arXiv:0901.3725 (2009).

    """

    measurement_model: MeasurementModel = Property(
        default=None,
        doc="A measurement model. This need not be defined if a measurement "
            "model is provided in the measurement. If no model specified on "
            "construction, or in the measurement, then error will be thrown. "
            )

    def _check_measurement_prediction(self, hypothesis, **kwargs):
        """Check to see if a measurement prediction exists in the hypothesis.

        Parameters
        ----------
        hypothesis : :class:`~.SingleHypothesis`
            the prediction-measurement association hypothesis. This hypothesis
            may carry a predicted measurement, or a predicted state. In the
            latter case a predicted measurement will be calculated.

        Returns
        -------
        hypothesis : :class:`~.SingleHypothesis`
            the prediction-measurement association hypothesis. This hypothesis
            may carry a predicted measurement, or a predicted state. In the
            latter case a predicted measurement will be calculated.
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
        return hypothesis

    def predict_measurement(self, predicted_state, measurement_model=None,
                            **kwargs):
        r"""Predict the measurement implied by the predicted state mean

        Parameters
        ----------
        pred_state : :class:`~.State`
            The predicted state :math:`\mathbf{x}_{k|k-1}`
        measurement_model : :class:`~.MeasurementModel`
            The measurement model. If omitted, the model in the updater object
            is used

        Returns
        -------
        : :class:`EnsembleMeasurementPrediction`
            The measurement prediction, :math:`\mathbf{z}_{k|k-1}`

        """
        # If a measurement model is not specified then use the one that's
        # native to the updater
        measurement_model = self._check_measurement_model(measurement_model)

        # Propagate each vector through the measurement model.
        pred_meas_ensemble = StateVectors(
                [measurement_model.function(State(state_vector=ensemble_member),
                                            noise=True)
                 for ensemble_member in predicted_state.ensemble.T])

        return MeasurementPrediction.from_state(
                   predicted_state, pred_meas_ensemble)

    def update(self, hypothesis, **kwargs):
        r"""The Ensemble Kalman update method. The Ensemble Kalman filter
        simply uses the Kalman Update scheme
        to evolve a set or Ensemble
        of state vectors as a group. This ensemble of vectors contains all the
        information on the system state.

        Parameters
        ----------
        hypothesis : :class:`~.SingleHypothesis`
            the prediction-measurement association hypothesis. This hypothesis
            may carry a predicted measurement, or a predicted state. In the
            latter case a predicted measurement will be calculated.


        Returns
        -------
        : :class:`~.EnsembleStateUpdate`
            The posterior state which contains an ensemble of state vectors
            and a timestamp.

        """
        # Assigning more readible variable names
        hypothesis = self._check_measurement_prediction(hypothesis)
        pred_state = hypothesis.prediction
        meas_mean = hypothesis.measurement.state_vector
        meas_covar = self.measurement_model.covar()
        num_vectors = hypothesis.prediction.num_vectors
        prior_ensemble = hypothesis.prediction.ensemble

        # Generate an ensemble of measurements based on measurement
        measurement_ensemble = hypothesis.prediction.generate_ensemble(
                                   mean=meas_mean,
                                   covar=meas_covar,
                                   num_vectors=num_vectors)

        # Calculate Kalman Gain according to Dr. Jan Mandel's EnKF formalism.
        innovation_ensemble = hypothesis.prediction.ensemble - hypothesis.prediction.mean

        meas_innovation = StateVectors([self.measurement_model.function(
            State(state_vector=col), noise=False) - self.measurement_model.function(
                State(state_vector=hypothesis.prediction.mean), noise=False)
                for col in prior_ensemble.T])

        # Calculate Kalman Gain
        kalman_gain = 1/(num_vectors-1) * innovation_ensemble @ meas_innovation.T @ \
            scipy.linalg.inv(1/(num_vectors-1) * meas_innovation @ meas_innovation.T + meas_covar)

        # Calculate Posterior Ensemble
        posterior_ensemble = pred_state.ensemble + \
            kalman_gain @ (measurement_ensemble -
                           hypothesis.measurement_prediction.ensemble)

        return Update.from_state(hypothesis.prediction,
                                 posterior_ensemble,
                                 timestamp=hypothesis.measurement.timestamp,
                                 hypothesis=hypothesis)
