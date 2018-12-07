# -*- coding: utf-8 -*-

import numpy as np

from .base import Updater
from ..types import (GaussianMeasurementPrediction,
                     GaussianStateUpdate)
from ..types import MissedDetection


class KalmanUpdater(Updater):
    """Simple Kalman Updater

    Perform measurement update step in the standard Kalman Filter.
    """

    def get_measurement_prediction(self, state_prediction, **kwargs):
        """Kalman Filter measurement prediction step

        Parameters
        ----------
        state_prediction : :class:`~.GaussianStatePrediction`
            A predicted state object

        Returns
        -------
        : :class:`~.GaussianMeasurementPrediction`
            The measurement prediction
        """

        # Measurement model parameters
        measurement_matrix = self.measurement_model.matrix(**kwargs)
        measurement_noise_covar = self.measurement_model.covar(**kwargs)

        meas_pred_mean, meas_pred_covar, cross_covar = \
            self.get_measurement_prediction_lowlevel(state_prediction.mean,
                                                     state_prediction.covar,
                                                     measurement_matrix,
                                                     measurement_noise_covar)

        return GaussianMeasurementPrediction(meas_pred_mean, meas_pred_covar,
                                             state_prediction.timestamp,
                                             cross_covar)

    def update(self, hypothesis, **kwargs):
        """Kalman Filter update step

        Parameters
        ----------
        hypothesis : :class:`~.Hypothesis`
            Hypothesis with predicted state and associated detection used for
            updating.

        Returns
        -------
        : :class:`~.GaussianStateUpaate`
            The computed state posterior
        """

        if hypothesis.measurement_prediction is None:
            hypothesis.measurement_prediction = \
                self.get_measurement_prediction(hypothesis.prediction,
                                                **kwargs)

        posterior_mean, posterior_covar, _ = \
            self._update_on_measurement_prediction(
                hypothesis.prediction.mean,
                hypothesis.prediction.covar,
                hypothesis.measurement.state_vector,
                hypothesis.measurement_prediction.mean,
                hypothesis.measurement_prediction.covar,
                hypothesis.measurement_prediction.cross_covar)

        return GaussianStateUpdate(posterior_mean,
                                   posterior_covar,
                                   hypothesis,
                                   hypothesis.measurement.timestamp)

    @staticmethod
    def update_lowlevel(x_pred, P_pred, H, R, y):
        """Low level Kalman Filter update

        Parameters
        ----------
        x_pred: :class:`numpy.ndarray` of shape (Ns,1)
            The predicted state mean
        P_pred: :class:`numpy.ndarray` of shape (Ns,Ns)
            The predicted state covariance
        H : :class:`numpy.ndarray` of shape (Nm,Ns)
            The measurement model matrix
        R : :class:`numpy.ndarray` of shape (Nm,Nm)
            The measurement noise covariance matrix
        y : :class:`numpy.ndarray` of shape (Nm,1)
            The measurement vector

        Returns
        -------
        : :class:`numpy.ndarray` of shape (Ns,1)
            The computed posterior state mean
        : :class:`numpy.ndarray` of shape (Ns,Ns)
            The computed posterior state covariance
        : :class:`numpy.ndarray` of shape (Ns,Nm)
            The computed Kalman gain
        """

        y_pred, S, Pxy = \
            KalmanUpdater.get_measurement_prediction_lowlevel(x_pred,
                                                              P_pred,
                                                              H, R)

        return KalmanUpdater._update_on_measurement_prediction(x_pred, P_pred,
                                                               y, y_pred, S,
                                                               Pxy)

    @staticmethod
    def get_measurement_prediction_lowlevel(x_pred, P_pred, H, R):
        """Low level Kalman Filter measurement prediction

        Parameters
        ----------
        x_pred: :class:`numpy.ndarray` of shape (Ns,1)
            The predicted state mean
        P_pred: :class:`numpy.ndarray` of shape (Ns,Ns)
            The predicted state covariance
        H : :class:`numpy.ndarray` of shape (Nm,Ns)
            The measurement model matrix
        R : :class:`numpy.ndarray` of shape (Nm,Nm)
            The measurement noise covariance matrix

        Returns
        -------
        : :class:`numpy.ndarray` of shape (Nm,1)
            The predicted measurement mean
        : :class:`numpy.ndarray` of shape (Nm,Nm)
            The predicted measurement noise (innovation) covariance matrix
        : :class:`numpy.ndarray` of shape (Ns,Nm), optional
            The state-to-measurement cross covariance
        """

        y_pred = H@x_pred
        S = H@P_pred@H.T + R
        Pxy = P_pred@H.T

        return y_pred, S, Pxy

    @staticmethod
    def _update_on_measurement_prediction(x_pred, P_pred, y,
                                          y_pred, S, Pxy):
        """Low level Kalman Filter update, based on a measurement prediction
        Parameters
        ----------
        x_pred: :class:`numpy.ndarray` of shape (Ns,1)
            The predicted state mean
        P_pred: :class:`numpy.ndarray` of shape (Ns,Ns)
            The predicted state covariance
        y : :class:`numpy.ndarray` of shape (Nm,1)
            The measurement vector
        y_pred: :class:`numpy.ndarray` of shape (Nm,1)
            The predicted measurement mean
        S: :class:`numpy.ndarray` of shape (Nm,Nm)
            The predicted measurement noise (innovation) covariance matrix
        Pxy: :class:`numpy.ndarray` of shape (Ns,Nm), optional
            The state-to-measurement cross covariance

        Returns
        -------
        : :class:`numpy.ndarray` of shape (Ns,1)
            The computed posterior state mean
        : :class:`numpy.ndarray` of shape (Ns,Ns)
            The computed posterior state covariance
        : :class:`numpy.ndarray` of shape (Ns,Nm)
            The computed Kalman gain
        """

        K = Pxy@np.linalg.inv(S)

        x_post = x_pred + K@(y-y_pred)
        P_post = P_pred - K@S@K.T

        return x_post, P_post, K


class PDAKalmanUpdater(Updater):
    """Probabilistic Data Association Kalman Updater

    Perform measurement update step in the Probabilistic Data
    Association (PDA) Kalman Filter.
    """

    def get_measurement_prediction(self, state_prediction, **kwargs):
        """Kalman Filter measurement prediction step

        Parameters
        ----------
        state_prediction : :class:`~.GaussianStatePrediction`
            A predicted state object

        Returns
        -------
        : :class:`~.GaussianMeasurementPrediction`
            The measurement prediction
        """

        # Measurement model parameters
        measurement_matrix = self.measurement_model.matrix(**kwargs)
        measurement_noise_covar = self.measurement_model.covar(**kwargs)

        meas_pred_mean, meas_pred_covar, cross_covar = \
            self.get_measurement_prediction_lowlevel(state_prediction.mean,
                                                     state_prediction.covar,
                                                     measurement_matrix,
                                                     measurement_noise_covar)

        return GaussianMeasurementPrediction(meas_pred_mean, meas_pred_covar,
                                             state_prediction.timestamp,
                                             cross_covar)

    def update(self, multihypothesis, **kwargs):
        """PDA Kalman Filter update step

        Parameters
        ----------
        multihypothesis : :class:`~.MultipleMeasurementHypothesis`
            MultipleMeasurementHypothesis with predicted state and associated
            detection used for updating.

        Returns
        -------
        : :class:`~.GaussianStateUpaate`
            The computed state posterior

        """

        if multihypothesis.measurement_prediction is None:
            multihypothesis.measurement_prediction = \
                self.get_measurement_prediction(multihypothesis.prediction,
                                                **kwargs)

        measurements = \
            np.hstack(measurement["measurement"].state_vector
                      for measurement in multihypothesis.weighted_measurements
                      if not isinstance(
                        measurement["measurement"], MissedDetection))
        weights = np.concatenate(
            (np.hstack([float(measurement["weight"])
                       for measurement in
                        multihypothesis.weighted_measurements
                       if isinstance(
                        measurement["measurement"], MissedDetection)]),
                np.hstack([float(measurement["weight"])
                          for measurement in
                           multihypothesis.weighted_measurements
                          if not isinstance(
                           measurement["measurement"], MissedDetection)])))
        posterior_mean, posterior_covar, _ = \
            self._update_on_measurement_prediction(
                multihypothesis.prediction.mean,
                multihypothesis.prediction.covar,
                measurements,
                weights,
                multihypothesis.measurement_prediction.mean,
                multihypothesis.measurement_prediction.covar,
                multihypothesis.measurement_prediction.cross_covar)

        return \
            GaussianStateUpdate(posterior_mean,
                                posterior_covar,
                                multihypothesis,
                                multihypothesis.weighted_measurements[-1]
                                ["measurement"].timestamp
                                )

    @staticmethod
    def update_lowlevel(x_pred, P_pred, H, R, y, W):
        """Low level Kalman Filter update

        Parameters
        ----------
        x_pred: :class:`numpy.ndarray` of shape (Ns,1)
            The predicted state mean
        P_pred: :class:`numpy.ndarray` of shape (Ns,Ns)
            The predicted state covariance
        H : :class:`numpy.ndarray` of shape (Nm,Ns)
            The measurement model matrix
        R : :class:`numpy.ndarray` of shape (Nm,Nm)
            The measurement noise covariance matrix
        y : :class:`numpy.ndarray` of shape (Nm,1)
            The measurement vector

        Returns
        -------
        : :class:`numpy.ndarray` of shape (Ns,1)
            The computed posterior state mean
        : :class:`numpy.ndarray` of shape (Ns,Ns)
            The computed posterior state covariance
        : :class:`numpy.ndarray` of shape (Ns,Nm)
            The computed Kalman gain
        """

        y_pred, S, Pxy = PDAKalmanUpdater.\
            get_measurement_prediction_lowlevel(x_pred, P_pred, H, R)

        return PDAKalmanUpdater.\
            _update_on_measurement_prediction(x_pred, P_pred, y, W,
                                              y_pred, S, Pxy)

    @staticmethod
    def get_measurement_prediction_lowlevel(x_pred, P_pred, H, R):
        """Low level Kalman Filter measurement prediction

        Parameters
        ----------
        x_pred: :class:`numpy.ndarray` of shape (Ns,1)
            The predicted state mean
        P_pred: :class:`numpy.ndarray` of shape (Ns,Ns)
            The predicted state covariance
        H : :class:`numpy.ndarray` of shape (Nm,Ns)
            The measurement model matrix
        R : :class:`numpy.ndarray` of shape (Nm,Nm)
            The measurement noise covariance matrix

        Returns
        -------
        : :class:`numpy.ndarray` of shape (Nm,1)
            The predicted measurement mean
        : :class:`numpy.ndarray` of shape (Nm,Nm)
            The predicted measurement noise (innovation) covariance matrix
        : :class:`numpy.ndarray` of shape (Ns,Nm), optional
            The state-to-measurement cross covariance
        """

        y_pred = H@x_pred
        S = H@P_pred@H.T + R
        Pxy = P_pred@H.T

        return y_pred, S, Pxy

    @staticmethod
    def _update_on_measurement_prediction(x_pred, P_pred, y, W,
                                          y_pred, S, Pxy):
        """Low level PDA Kalman Filter update, based on a measurement prediction
        Parameters
        ----------
        x_pred: :class:`numpy.ndarray` of shape (Ns,1)
            The predicted state mean
        P_pred: :class:`numpy.ndarray` of shape (Ns,Ns)
            The predicted state covariance
        y : :class:`numpy.ndarray` of shape (Nm,num_measurements)
            The measurement vector
        W: :class:`numpy.ndarray` of shape (Nm,num_measurements)
            The mixture weights vector
        y_pred: :class:`numpy.ndarray` of shape (Nm,num_measurements)
            The predicted measurement mean
        S: :class:`numpy.ndarray` of shape (Nm,Nm)
            The predicted measurement noise (innovation) covariance matrix
        Pxy: :class:`numpy.ndarray` of shape (Ns,Nm), optional
            The state-to-measurement cross covariance

        Returns
        -------
        : :class:`numpy.ndarray` of shape (Ns,1)
            The computed posterior state mean
        : :class:`numpy.ndarray` of shape (Ns,Ns)
            The computed posterior state covariance
        : :class:`numpy.ndarray` of shape (Ns,Nm)
            The computed Kalman gain
        """

        # compute the Kalman gain
        K = Pxy@np.linalg.inv(S)

        # compute innovation mean and (cross) covariance
        innov_err = y-y_pred
        tot_innov_err = innov_err@W[1:, None]  # W[1:, None] replaces W[1:].T
        Pc = P_pred - K@S@K.T
        Pgag = K @ ((innov_err*W[1:])@innov_err.T -
                    tot_innov_err @ tot_innov_err.T) @ K.T

        x_post = x_pred + K@tot_innov_err
        P_post = W[0]*P_pred + (1-W[0])*Pc + Pgag

        return x_post, P_post, K


class ExtendedKalmanUpdater(KalmanUpdater):
    """Extended Kalman Updater

    Perform measurement update step in the Extended Kalman Filter.
    """

    def get_measurement_prediction(self, state_prediction, **kwargs):
        """Extended Kalman Filter measurement prediction step

        Parameters
        ----------
        state_prediction : :class:`~.GaussianStatePrediction`
            A predicted state object

        Returns
        -------
        : :class:`~.GaussianMeasurementPrediction`
            The measurement prediction
        """

        # Measurement model parameters
        try:
            # Attempt to extract matrix from a LinearModel
            measurement_matrix = self.measurement_model.matrix(**kwargs)
        except AttributeError:
            # Else read jacobian from a NonLinearModel
            measurement_matrix = \
                self.measurement_model.jacobian(state_prediction.state_vector,
                                                **kwargs)
        measurement_noise_covar = self.measurement_model.covar(**kwargs)

        meas_pred_mean, meas_pred_covar, cross_covar = \
            self.get_measurement_prediction_lowlevel(state_prediction.mean,
                                                     state_prediction.covar,
                                                     measurement_matrix,
                                                     measurement_noise_covar)

        return GaussianMeasurementPrediction(meas_pred_mean, meas_pred_covar,
                                             state_prediction.timestamp,
                                             cross_covar)

    def update(self, hypothesis, **kwargs):
        """ Extended Kalman Filter update step

        Parameters
        ----------
        hypothesis : :class:`~.Hypothesis`
            Hypothesis with predicted state and associated detection used for
            updating.

        Returns
        -------
        : :class:`~.GaussianState`
            The state posterior
        """

        if hypothesis.measurement_prediction is None:
            hypothesis.measurement_prediction = \
                self.get_measurement_prediction(hypothesis.prediction,
                                                **kwargs)

        posterior_mean, posterior_covar, _ = \
            self._update_on_measurement_prediction(
                hypothesis.prediction.mean,
                hypothesis.prediction.covar,
                hypothesis.measurement.state_vector,
                hypothesis.measurement_prediction.mean,
                hypothesis.measurement_prediction.covar,
                hypothesis.measurement_prediction.cross_covar)

        return GaussianStateUpdate(posterior_mean,
                                   posterior_covar,
                                   hypothesis,
                                   hypothesis.measurement.timestamp)

    @staticmethod
    def update_lowlevel(x_pred, P_pred, H, R, y):
        """Low level Extended Kalman Filter update

        Parameters
        ----------
        x_pred: :class:`numpy.ndarray` of shape (Ns,1)
            The predicted state mean
        P_pred: :class:`numpy.ndarray` of shape (Ns,Ns)
            The predicted state covariance
        H : :class:`numpy.ndarray` of shape (Nm,Ns)
            The measurement model jacobian matrix
        R : :class:`numpy.ndarray` of shape (Nm,Nm)
            The measurement noise covariance matrix
        y : :class:`numpy.ndarray` of shape (Nm,1)
            The measurement vector

        Returns
        -------
        : :class:`numpy.ndarray` of shape (Ns,1)
            The computed posterior state mean
        : :class:`numpy.ndarray` of shape (Ns,Ns)
            The computed posterior state covariance
        : :class:`numpy.ndarray` of shape (Ns,Nm)
            The computed Kalman gain
        """

        return KalmanUpdater.update_lowlevel(x_pred, P_pred, H, R, y)

    @staticmethod
    def get_measurement_prediction_lowlevel(x_pred, P_pred, H, R):
        """Low level Extended Kalman Filter measurement prediction

        Parameters
        ----------
        x_pred: :class:`numpy.ndarray` of shape (Ns,1)
            The predicted state mean
        P_pred: :class:`numpy.ndarray` of shape (Ns,Ns)
            The predicted state covariance
        H : :class:`numpy.ndarray` of shape (Nm,Ns)
            The measurement model jacobian matrix
        R : :class:`numpy.ndarray` of shape (Nm,Nm)
            The measurement noise covariance matrix

        Returns
        -------
        : :class:`numpy.ndarray` of shape (Nm,1)
            The predicted measurement mean
        : :class:`numpy.ndarray` of shape (Nm,Nm)
            The predicted measurement noise (innovation) covariance matrix
        : :class:`numpy.ndarray` of shape (Ns,Nm), optional
            The state-to-measurement cross covariance
        """

        return KalmanUpdater.get_measurement_prediction_lowlevel(x_pred,
                                                                 P_pred,
                                                                 H, R)

    @staticmethod
    def _update_on_measurement_prediction(x_pred, P_pred, y,
                                          y_pred, S, Pxy):
        """Low level Extended Kalman Filter update, based on a measurement\
        prediction

        Parameters
        ----------
        x_pred: :class:`numpy.ndarray` of shape (Ns,1)
            The predicted state mean
        P_pred: :class:`numpy.ndarray` of shape (Ns,Ns)
            The predicted state covariance
        y : :class:`numpy.ndarray` of shape (Nm,1)
            The measurement vector
        y_pred: :class:`numpy.ndarray` of shape (Nm,1)
            The predicted measurement mean
        S: :class:`numpy.ndarray` of shape (Nm,Nm)
            The predicted measurement noise (innovation) covariance matrix
        Pxy: :class:`numpy.ndarray` of shape (Ns,Nm), optional
            The state-to-measurement cross covariance

        Returns
        -------
        : :class:`numpy.ndarray` of shape (Ns,1)
            The computed posterior state mean
        : :class:`numpy.ndarray` of shape (Ns,Ns)
            The computed posterior state covariance
        : :class:`numpy.ndarray` of shape (Ns,Nm)
            The computed Kalman gain
        """

        return KalmanUpdater._update_on_measurement_prediction(x_pred, P_pred,
                                                               y, y_pred, S,
                                                               Pxy)
