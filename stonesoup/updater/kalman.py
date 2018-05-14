# -*- coding: utf-8 -*-

import numpy as np

from .base import Updater
from ..types.state import GaussianState


class KalmanUpdater(Updater):
    """Simple Kalman Updater

    Perform measurement update step in the standard Kalman Filter.
    """

    def update(self, state_pred, meas_pred, meas, cross_covar=None, **kwargs):
        """Kalman Filter update step

        Parameters
        ----------
        state_pred : :class:`GaussianState`
            The state prediction
        meas_pred : :class:`GaussianState`
            The measurement prediction
        meas : :class:`Detection`
            The measurement
        cross_covar: :class:`numpy.ndarray` of shape (Nm,Nm), optional
            The state-to-measurement cross covariance (the default is None, in
            which case ``cross_covar`` will be computed internally)

        Returns
        -------
        :class:`GaussianState`
            The state posterior
        :class:`numpy.ndarray` of shape (Ns,Nm)
            The computed Kalman gain matrix
        """

        if(cross_covar is None):
            cross_covar = state_pred.covar@self.measurement_model.matrix(
                **kwargs).T

        state_post_mean, state_post_covar, kalman_gain = \
            self._update(state_pred.mean, state_pred.covar, meas.state_vector,
                         meas_pred.mean, meas_pred.covar, cross_covar)

        state_post = GaussianState(state_post_mean,
                                   state_post_covar,
                                   meas_pred.timestamp)

        return state_post, kalman_gain

    @staticmethod
    def _update(x_pred, P_pred, y, y_pred, S, Pxy):
        """Low level Kalman Filter update

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
        Pxy: :class:`numpy.ndarray` of shape (Nm,Nm)
            The calculated state-to-measurement cross covariance
        Returns
        -------
        x_post: :class:`numpy.ndarray` of shape (Ns,1)
            The computed posterior state mean
        P_post: :class:`numpy.ndarray` of shape (Ns,Ns)
            The computed posterior state covariance
        K: :class:`numpy.ndarray` of shape (Ns,Nm)
            The computed Kalman gain
        """

        K = Pxy@np.linalg.inv(S)

        x_post = x_pred + K@(y-y_pred)
        P_post = P_pred - K@S@K.T

        return x_post, P_post, K


class ExtendedKalmanUpdater(KalmanUpdater):
    """Extended Kalman Updater

    Perform measurement update step in the Extended Kalman Filter.
    """

    def update(self, state_pred, meas_pred, meas, cross_covar=None, **kwargs):
        """ExtendedKalman Filter update step

        Parameters
        ----------
        state_pred : :class:`GaussianState`
            The state prediction
        meas_pred : :class:`GaussianState`
            The measurement prediction
        meas : :class:`numpy.ndarray` of shape (Nm,1)
            The measurement vector
        cross_covar: :class:`numpy.ndarray` of shape (Nm,Nm), optional
            The state-to-measurement cross covariance (the default is None, in
            which case ``cross_covar`` will be computed internally)

        Returns
        -------
        :class:`GaussianState`
            The state posterior
        :class:`numpy.ndarray` of shape (Ns,Nm)
            The computed Kalman gain
        """

        if(cross_covar is None):
            # Measurement model parameters
            try:
                # Attempt to extract matrix from a LinearModel
                measurement_matrix = self.measurement_model.matrix(**kwargs)
            except AttributeError:
                # Else read jacobian from a NonLinearModel
                measurement_matrix = self.measurement_model.jacobian(**kwargs)
            cross_covar = state_pred.covar@measurement_matrix.T

        state_post_mean, state_post_covar, kalman_gain = \
            super()._update(state_pred.mean, state_pred.covar,
                            meas.state_vector, meas_pred.mean,
                            meas_pred.covar, cross_covar)

        state_post = GaussianState(state_post_mean,
                                   state_post_covar,
                                   meas_pred.timestamp)

        return state_post, kalman_gain
