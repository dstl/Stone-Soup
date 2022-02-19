# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg

from .kalman import KalmanUpdater
from ..base import Property
from ..types.state import State
from ..types.array import StateVectors
from ..types.prediction import MeasurementPrediction
from ..types.update import Update, EnsembleStateUpdate
from ..models.measurement import MeasurementModel

class EnsembleUpdater(KalmanUpdater):
    r"""Ensemble Kalman Filter Updater class

    The EnKF is a hybrid of the Kalman updating scheme and the 
    Monte Carlo aproach of the the particle filter.
    
    Deliberatly structured to resemble the Vanilla Kalman Filter,
    :meth:`update` first calls :meth:`predict_measurement` function which
    proceeds by calculating the predicted measurement, innovation covariance
    and measurement cross-covariance,

    .. math::

        \mathbf{z}_{k|k-1} = H_k \mathbf{x}_{k|k-1}

        S_k = H_k P_{k|k-1} H_k^T + R_k

        Upsilon_k = P_{k|k-1} H_k^T

    where :math:`P_{k|k-1}` is the predicted state covariance.
    :meth:`predict_measurement` returns a
    :class:`~.EnsembleMeasurementPrediction`. The Kalman gain is then
    calculated identically as in the Linear Kalman Filter, however the mean
    of the ensemble is used as the representation of the system's state prediction
    covariance.
    """

    measurement_model: MeasurementModel = Property(
        default=None,
        doc="A measurement model. This need not be defined if a measurement "
            "model is provided in the measurement. If no model specified on "
            "construction, or in the measurement, then error will be thrown. "
            )

    def _check_measurement_prediction(self, hypothesis,**kwargs):    
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
        
        # Get the predicted state from the hypothesis
        predicted_state = hypothesis.prediction
        # If there is no measurement prediction in the hypothesis then do the
        # measurement prediction (and attach it back to the hypothesis).
        if hypothesis.measurement_prediction is None:
            # Get the measurement model out of the measurement if it's there.
            # If not, use the one native to the updater 
            measurement_model = hypothesis.measurement.measurement_model
            measurement_model = self._check_measurement_model(measurement_model)
            # Attach the measurement prediction to the hypothesis
            hypothesis.measurement_prediction = self.predict_measurement(
                predicted_state=predicted_state, 
                measurement_model=measurement_model, **kwargs)
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

        pred_meas_ensemble = StateVectors([measurement_model.function(State(state_vector=ensemble_member),
                                      noise=True) for ensemble_member in predicted_state.ensemble.T])

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
        #More readible variable names
        hypothesis = self._check_measurement_prediction(hypothesis)
        pred_state = hypothesis.prediction
        meas_mean = hypothesis.measurement.state_vector
        meas_covar = self.measurement_model.covar()
        num_vectors = hypothesis.prediction.num_vectors
        prior_ensemble = hypothesis.prediction.ensemble
        
        #Generate an ensemble of measurements based on measurement
        measurement_ensemble = hypothesis.prediction.generate_ensemble(mean=meas_mean,
                                                                       covar=meas_covar,
                                                                       num_vectors=num_vectors)
        
        #Calculate Kalman Gain according to Jan Mandel's EnKF formulation
        A = hypothesis.prediction.ensemble - hypothesis.prediction.mean

        HA = StateVectors([self.measurement_model.function(
            State(state_vector=col),noise=False) - self.measurement_model.function(
                State(state_vector=hypothesis.prediction.mean),noise=False)
            for col in prior_ensemble.T])
        #Calculate Kalman Gain
        kalman_gain = 1/(num_vectors-1) * A @ HA.T @ np.linalg.inv(
            1/(num_vectors-1)* HA @ HA.T+meas_covar)

        posterior_ensemble = pred_state.ensemble + \
                            kalman_gain@(measurement_ensemble -
                            hypothesis.measurement_prediction.ensemble)

        return Update.from_state(hypothesis.prediction,
                    posterior_ensemble,timestamp=hypothesis.measurement.timestamp,
                    hypothesis=hypothesis)
    
class EnsembleSqrtUpdater(EnsembleUpdater):
    r"""Polynomial Chaos Expansion based Ensemble Kalman Filter Updater class

    The PCEnKF (Polynomial Chaos Ensemble Kalman Filter) quantifies the state
    with a Polynomial Chaos Expansion, a sum of orthogonal Hermite Polynomials
    and associated coefficients.
    
    The coefficients of these expansions allow for easy computation of the mean
    and covariance for an ensemble of state vectors. A new ensemble can then be
    sampled from this mean and covariance specified by the coefficients.
    
    This updater class uses the EnKF updating routine to obtain the posterior
    ensemble, then the corresponding PCE Expansion is computed for that ensemble.
    """
    
    def update(self, hypothesis, **kwargs):
        r"""The Ensemble Kalman update method. The Ensemble Square Root filter
        propagates the mean and square root covariance through time, and samples
        a new ensemble. This has the advantage of not peturbing the measurement
        with statistical noise, and thus is less prone to sampling error for 
        small ensembles.

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
        #More readible variable names
        hypothesis = self._check_measurement_prediction(hypothesis)
        pred_state = hypothesis.prediction
        num_vectors=hypothesis.prediction.num_vectors
        measurement = hypothesis.measurement.state_vector      
        meas_covar = self.measurement_model.covar()
        
        #Calculate Kalman Gain     
        cross_covar = pred_state.sqrt_covar @ hypothesis.measurement_prediction.sqrt_covar.T
        innovation_covar = hypothesis.measurement_prediction.sqrt_covar @ hypothesis.measurement_prediction.sqrt_covar.T + meas_covar
        kalman_gain = cross_covar @ np.linalg.inv(innovation_covar)
        
        #Calculate posterior sqrt covar
        posterior_sqrt_covar = pred_state.sqrt_covar @ \
                ((np.eye(num_vectors) - 
                hypothesis.measurement_prediction.sqrt_covar.T @ 
                np.linalg.inv(innovation_covar) @
                hypothesis.measurement_prediction.sqrt_covar))
        
        #Calculate new mean and covar, then sample new ensemble
        posterior_mean = pred_state.state_vector + \
                         kalman_gain@(measurement - 
                                      hypothesis.measurement_prediction.state_vector)
        posterior_covar = posterior_sqrt_covar @ posterior_sqrt_covar.T
        
        posterior_ensemble = pred_state.generate_ensemble(posterior_mean,
                                                          posterior_covar,
                                                          hypothesis.prediction.num_vectors)
        
        return Update.from_state(hypothesis.prediction,
                    posterior_ensemble,timestamp=hypothesis.measurement.timestamp,
                    hypothesis=hypothesis)


