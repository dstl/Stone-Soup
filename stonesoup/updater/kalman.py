# -*- coding: utf-8 -*-

import numpy as np

from .base import Updater
from ..functions import tria, jacobian
from ..measurementmodel import MeasurementModel
from ..types.base import GaussianState, StateVector
from ..base import Property


class KalmanUpdater(Updater):
    """Simple Kalman Filter

    Perform measurement update step in the standard Kalman Filter.

    Parameters
    ----------
    meas_model : :class:`MeasurementModel`
        The measurement model
    """

    meas_model = Property(MeasurementModel, doc="measurement model")

    def __init__(self, meas_model, *args, **kwargs):
        """Constructor method

        Parameters
        ----------
        meas_model : :class:`MeasurementModel`
            The measurement model
        """

        super().__init__(meas_model, *args, **kwargs)

    def update(self, state_pred, meas_pred, meas, cross_covar=None):
        """Kalman Filter update step

        Parameters
        ----------
        state_pred : :class:`GaussianState`
            The state prediction
        meas_pred : :class:`GaussianState`
            The measurement prediction
        meas : 1-D numpy.ndarray of shape (Nm,1)
            The measurement vector
        cross_covar: 2-D numpy.ndarray of shape (Nm,Nm), optional
            The state-to-measurement cross covariance (the default is None, in
            which case ``cross_covar`` will be computed internally)

        Returns
        -------
        state_post : :class:`GaussianState`
            The state posterior
        kalman_gain : 2-D numpy.ndarray of shape (Ns,Nm)
            The computed Kalman gain
        """

        if(cross_covar is None):
            cross_covar = state_pred.covar@self.meas_model.eval().T

        state_post = GaussianState()

        state_post.mean, state_post.covar, kalman_gain = \
            self._update(state_pred.mean, state_pred.covar, meas,
                         meas_pred.mean, meas_pred.covar, cross_covar)

        return state_post, kalman_gain

    @staticmethod
    def _update(x_pred, P_pred, y, y_pred, S, Pxy):
        """Low level Kalman Filter update

        Parameters
        ----------
        x_pred: 1-D numpy.ndarray of shape (Ns,1)
            The predicted state mean
        P_Pred: 2-D numpy.ndarray of shape (Ns,Ns)
            The predicted state covariance
        y : 1-D numpy.ndarray of shape (Nm,1)
            The measurement vector
        y_pred: 1-D numpy.ndarray of shape (Nm,1)
            The predicted measurement mean
        S: 2-D numpy.ndarray of shape (Nm,Nm)
            The predicted measurement noise (innovation) covariance matrix
        Pxy: 2-D numpy.ndarray of shape (Nm,Nm)
            The calculated state-to-measurement cross covariance
        Returns
        -------
        x_post: 1-D numpy.ndarray of shape (Ns,1)
            The computed posterior state mean
        P_post: 2-D numpy.ndarray of shape (Ns,Ns)
            The computed posterior state covariance
        K: 2-D numpy.ndarray of shape (Ns,Nm)
            The computed Kalman gain
        """

        K = Pxy@np.linalg.inv(S)

        x_post = x_pred + K@(y-y_pred)
        P_post = P_pred - K@S@K.T

        return x_post, P_post, K


class ExtendedKalmanUpdater(KalmanUpdater):
    """Extended Kalman Filter

    Perform measurement update step in the Extended Kalman Filter.

    Parameters
    ----------
    meas_model : :class:`MeasurementModel`
        The measurement model
    """

    meas_model = Property(MeasurementModel, doc="measurement model")

    def __init__(self, meas_model, *args, **kwargs):
        """Constructor method

        Parameters
        ----------
        meas_model : :class:`MeasurementModel`
            The measurement model
        """

        super().__init__(meas_model, *args, **kwargs)

    def update(self, state_pred, meas_pred, meas, cross_covar=None):
        """ExtendedKalman Filter update step

        Parameters
        ----------
        state_pred : :class:`GaussianState`
            The state prediction
        meas_pred : :class:`GaussianState`
            The measurement prediction
        meas : 1-D numpy.ndarray of shape (Nm,1)
            The measurement vector
        cross_covar: 2-D numpy.ndarray of shape (Nm,Nm), optional
            The state-to-measurement cross covariance (the default is None, in
            which case ``cross_covar`` will be computed internally)

        Returns
        -------
        state_post : :class:`GaussianState`
            The state posterior
        kalman_gain : 2-D numpy.ndarray of shape (Ns,Nm)
            The computed Kalman gain
        """

        if(cross_covar is None):
            def h(x):
                return self.meas_model.eval(x)
            H = jacobian(h, state_pred.mean)
            cross_covar = state_pred.covar@H.T

        state_post = GaussianState()

        state_post.mean, state_post.covar, kalman_gain = \
            super()._update(state_pred.mean, state_pred.covar, meas,
                            meas_pred.mean, meas_pred.covar, cross_covar)

        return state_post, kalman_gain


class SqrtKalmanUpdater(Updater):
    """Square Root Kalman Filter

    Perform measurement update step in the square root Kalman Filter.
    """

    @staticmethod
    def update(track, detection, meas_mat=None):
        # track.covar and detection.covar are lower triangular matrices
        if meas_mat is None:
            meas_mat = np.eye(len(detection.state), len(track.state))

        innov = detection.state - meas_mat @ track.state

        Pxz = track.covar @ track.covar.T @ meas_mat.T
        innov_covar = tria(np.concatenate(
            ((meas_mat @ track.covar), detection.covar),
            axis=1))
        gain = Pxz @ np.linalg.inv(innov_covar.T) @ np.linalg.inv(innov_covar)

        updated_state = track.state + gain @ innov

        temp = gain @ meas_mat
        updated_state_covar = tria(np.concatenate(
            (((np.eye(*temp.shape) - temp) @ track.covar),
             (gain @ detection.covar)),
            axis=1))

        return (
            StateVector(updated_state, updated_state_covar),
            StateVector(innov, innov_covar))
