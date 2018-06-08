# -*- coding: utf-8 -*-

import numpy as np

from .base import Updater
from ..types.state import GaussianState


class KalmanUpdater(Updater):
    """Simple Kalman Updater

    Perform measurement update step in the standard Kalman Filter.
    """

    def update(self, prediction, measurement, **kwargs):
        """Kalman Filter update step

        Parameters
        ----------
        prediction : :class:`GaussianState`
            The state prediction
        measurement : :class:`Detection`
            The measurement

        Returns
        -------
        :class:`GaussianState`
            The state posterior
        """

        # Measurement model parameters
        measurement_matrix = self.measurement_model.matrix(**kwargs)
        measurement_noise_covar = self.measurement_model.covar(**kwargs)

        posterior_mean, posterior_covar, _ = \
            self.update_lowlevel(prediction.mean, prediction.covar,
                                 measurement_matrix, measurement_noise_covar,
                                 measurement.state_vector)

        posterior = GaussianState(posterior_mean,
                                  posterior_covar,
                                  prediction.timestamp)

        return posterior

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
        x_post: :class:`numpy.ndarray` of shape (Ns,1)
            The computed posterior state mean
        P_post: :class:`numpy.ndarray` of shape (Ns,Ns)
            The computed posterior state covariance
        K: :class:`numpy.ndarray` of shape (Ns,Nm)
            The computed Kalman gain
        """

        y_pred = H@x_pred
        S = H@P_pred@H.T + R
        Pxy = P_pred@H.T
        K = Pxy@np.linalg.inv(S)
        x_post = x_pred + K@(y-y_pred)
        P_post = P_pred - K@S@K.T

        return x_post, P_post, K


class ExtendedKalmanUpdater(KalmanUpdater):
    """Extended Kalman Updater

    Perform measurement update step in the Extended Kalman Filter.
    """

    def update(self, prediction, measurement, **kwargs):
        """ Extended Kalman Filter update step

        Parameters
        ----------
        prediction : :class:`GaussianState`
            The state prediction
        measurement : :class:`Detection`
            The measurement

        Returns
        -------
        :class:`GaussianState`
            The state posterior
        """

        # Measurement model parameters
        try:
            # Attempt to extract matrix from a LinearModel
            measurement_matrix = self.measurement_model.matrix(**kwargs)
        except AttributeError:
            # Else read jacobian from a NonLinearModel
            measurement_matrix = self.measurement_model.jacobian(**kwargs)
        measurement_noise_covar = self.measurement_model.covar(**kwargs)

        posterior_mean, posterior_covar, _ = \
            self.update_lowlevel(prediction.mean,
                                 prediction.covar,
                                 measurement_matrix,
                                 measurement_noise_covar,
                                 measurement.state_vector)

        posterior = GaussianState(posterior_mean,
                                  posterior_covar,
                                  prediction.timestamp)

        return posterior

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
        x_post: :class:`numpy.ndarray` of shape (Ns,1)
            The computed posterior state mean
        P_post: :class:`numpy.ndarray` of shape (Ns,Ns)
            The computed posterior state covariance
        K: :class:`numpy.ndarray` of shape (Ns,Nm)
            The computed Kalman gain
        """

        return [*KalmanUpdater.update_lowlevel(x_pred, P_pred, H, R, y)]
