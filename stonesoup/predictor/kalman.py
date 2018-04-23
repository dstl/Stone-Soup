# -*- coding: utf-8 -*-

import numpy as np

from ..transitionmodel.base import TransitionModel
from ..measurementmodel.base import MeasurementModel
from ..controlmodel.base import ControlModel
from .base import Predictor
from ..types.base import GaussianState
from ..base import Property


class KalmanPredictor(Predictor):
    """KalmanPredictor class

    An implementation of a standard Kalman Filter predictor.

    Parameters
    ----------
    trans_model : :class:`TransitionModel`
        The transition model used to perform the state prediction
    meas_model : :class:`MeasurementModel`
        The measurement model used to generate the measurement prediction
    ctrl_model : :class:`ControlModel`
        The (optional) control model used during the state prediction
    """

    trans_model = Property(TransitionModel, doc="transition model")
    meas_model = Property(MeasurementModel, doc="measurement model")
    ctrl_model = Property(ControlModel, doc="control model")

    def __init__(self, trans_model, meas_model=None,
                 ctrl_model=None, *args, **kwargs):
        """Constructor method

        trans_model : :class:`TransitionModel`
            The transition model used to perform the state prediction
        meas_model : :class:`MesurementModel`, optional
            The measurement model used to generate the measurement prediction
            (the default is None)
        ctrl_model : :class:`ControlModel`, optional
            The (optional) control model used during the state prediction (the
            default is None)

        """

        # TODO: Input validation

        super().__init__(trans_model, meas_model,
                         ctrl_model, *args, **kwargs)

    def predict(self, state_prior, ctrl_input=None):
        """Kalman Filter prediction step

        state_prior : :class:`GaussianState`
            A prior state object
        ctrl_input : 1-D array of shape (Nu,1), optional
            The control input vector. It will only have an effect if
            :attr:`ctrl_model` is not None

        Returns
        -------
        state_pred : :class:`GaussianState`
            The state prediction
        meas_pred : :class:`GaussianState`
            The measurement prediction
        cross_covar: 2-D numpy.ndarray of shape (Nm,Nm)
            The calculated state-to-measurement cross covariance
        """

        state_pred = self.predict_state(state_prior, ctrl_input)
        meas_pred, cross_covar = self.predict_meas(state_pred)

        return (state_pred, meas_pred, cross_covar)

    def predict_state(self, state_prior, ctrl_input=None):
        """Kalman Filter state prediction step

        state_prior : :class:`GaussianState`
            A prior state object
        ctrl_input : 1-D array of shape (Nu,1), optional
            The control input vector. It will only have an effect if
            :attr:`ctrl_model` is not None

        Returns
        -------
        state_pred : :class:`GaussianState`
            The state prediction
        """

        # TODO: Input validation

        x_prior = state_prior.mean
        P_prior = state_prior.covar
        F = self.trans_model.eval()
        Q = self.trans_model.covar()

        if self.ctrl_model is not None:
            B = self.ctrl_model.eval()
            Qu = self.ctrl_model.covar()
        else:
            B = np.ones((self.trans_model.ndim_state, 2))
            Qu = np.zeros(2)

        if ctrl_input is None:
            u = np.zeros((2, 1))
        else:
            u = ctrl_input

        # Perform state prediction
        state_pred = GaussianState()
        state_pred.mean, state_pred.covar = self._predict_state(
            x_prior, P_prior, F, Q, u, B, Qu)

        return state_pred

    def predict_meas(self, state_pred=None):
        """Kalman Filter measurement prediction step

        state_pred : :class:`GaussianState`, optional
            A state prediction object

        Returns
        -------
        meas_pred : :class:`GaussianState`
            The measurement prediction
        cross_covar : 2-D numpy.ndarray of shape (Nm,Nm)
            The predicted measurement noise (innovation) covariance matrix
        """

        # TODO: Input validation

        x_pred = state_pred.mean
        P_pred = state_pred.covar
        H = self.meas_model.eval()
        R = self.meas_model.covar()

        meas_pred = GaussianState()
        meas_pred.mean, meas_pred.covar, cross_covar = \
            self._predict_meas(x_pred, P_pred, H, R)

        return meas_pred, cross_covar

    @staticmethod
    def _predict_state(x, P, F, Q, u, B, Qu):
        """Low-level Kalman Filter state prediction

        Parameters:
        ----------
        x : 1-D numpy.ndarray of shape (Ns,1)
            The prior state mean
        P : 2-D numpy.ndarray of shape (Ns,Ns)
            The prior state covariance
        F : 2-D numpy.ndarray of shape (Ns,Ns)
            The state transition matrix
        Q : 2-D numpy.ndarray of shape (Ns,Ns)
            The process noise covariance matrix
        u : 1-D numpy.ndarray of shape (Nu,1)
            The control input
        B : 2-D numpy.ndarray of shape (Ns,Nu)
            The control gain matrix
        Qu : 2-D numpy.ndarray of shape (Ns,Ns)
            The control process covariance matrix

        Returns
        -------
        x_pred: 1-D numpy.ndarray of shape (Ns,1)
            The predicted state mean
        P_Pred: 2-D numpy.ndarray of shape (Ns,Ns)
            The predicted state covariance
        """

        x_pred = F@x + B@u
        P_pred = F@P@F.T + Q + B@Qu@B.T

        return x_pred, P_pred

    @staticmethod
    def _predict_meas(x_pred, P_pred, H, R):
        """Low-level Kalman Filter measurement prediction

        Parameters:
        ----------
        x_pred : 1-D numpy.ndarray of shape (Ns,1)
            The predicted state mean
        P_pred : 2-D numpy.ndarray of shape (Ns,Ns)
            The predicted state covariance
        H : 2-D numpy.ndarray of shape (Nm,Ns)
            The measurement model matrix
        R : 2-D numpy.ndarray of shape (Nm,Nm)
            The measurement noise covariance matrix

        Returns
        -------
        y_pred: 1-D numpy.ndarray of shape (Nm,1)
            The predicted measurement mean
        S: 2-D numpy.ndarray of shape (Nm,Nm)
            The predicted measurement noise (innovation) covariance matrix
        Pxy: 2-D numpy.ndarray of shape (Nm,Nm)
            The calculated state-to-measurement cross-covariance
        """

        y_pred = H@x_pred
        S = H@P_pred@H.T + R
        Pxy = P_pred@H.T

        return y_pred, S, Pxy
