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
    state_prior : :class:`GaussianState`
        The prior state
    state_pred : :class:`GaussianState`
        The predicted state
    meas_pred : :class:`GaussianState`
        The predicted measurement
    trans_model : :class:`TransitionModel`
        The transition model used to perform the state prediction
    meas_model :
        The measurement model used to generate the measurement prediction
    ctrl_model : :class:`ControlModel`
        The (optional) control model used during the state prediction
    """

    trans_model = Property(TransitionModel, doc="transition model")
    state_prior = Property(GaussianState, doc="state prior")
    meas_model = Property(MeasurementModel, doc="measurement model")
    ctrl_model = Property(ControlModel, doc="control model")

    def __init__(self, trans_model, state_prior=None,
                 meas_model=None, ctrl_model=None, *args, **kwargs):
        """Constructor method

        trans_model : :class:`TransitionModel`
            The transition model used to perform the state prediction
        state_prior : :class:`GaussianState`, optional
            The prior state (the default is None)
        meas_model : :class:`MesurementModel`, optional
            The measurement model used to generate the measurement prediction
            (the default is None)
        ctrl_model : :class:`ControlModel`, optional
            The (optional) control model used during the state prediction (the
            default is None)

        """

        # TODO: Input validation

        super().__init__(trans_model, state_prior, meas_model,
                         ctrl_model, *args, **kwargs)

    def predict(self, ctrl_input=None, state_prior=None):
        """Kalman Filter prediction step

        ctrl_input : 1-D array of shape (Nu,1), optional
            The control input vector. It will only have an effect if
            :attr:`ctrl_model` is not None
        state_prior : :class:`GaussianState`, optional
            An "ad-hoc"prior state object (the default is None, which
            implies that state will be extracted internally from the object)

        Returns
        -------
        state_pred : :class:`GaussianState`
            The state prediction
        meas_pred : :class:`GaussianState`
            The measurement prediction
        """

        self.predict_state(ctrl_input=ctrl_input, state_prior=state_prior)
        self.predict_meas()

        return (self.state_pred, self.meas_pred)

    def predict_state(self, ctrl_input=None, state_prior=None):
        """Kalman Filter state prediction step

        ctrl_input : 1-D array of shape (Nu,1), optional
            The control input vector. It will only have an effect if
            :attr:`ctrl_model` is not None
        state_prior : :class:`GaussianState`, optional
            An "ad-hoc"prior state object (the default is None, which
            implies that state will be extracted internally from the object)

        Returns
        -------
        state_pred : :class:`GaussianState`
            The state prediction
        """

        # TODO: Input validation

        if state_prior is not None:
            self.state_prior = state_prior

        x_prior = self.state_prior.mean
        P_prior = self.state_prior.covar
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
        self.state_pred = GaussianState()
        self.state_pred.mean, self.state_pred.covar = self._predict_state(
            x_prior, P_prior, F, Q, u, B, Qu)

        return self.state_pred

    def predict_meas(self, state_pred=None):
        """Kalman Filter measurement prediction step

        state_pred : :class:`GaussianState`, optional
            An "ad-hoc" state prediction object (the default is None, which
            implies that state will be extracted internally from the object)

        Returns
        -------
        meas_pred : :class:`GaussianState`
            The measurement prediction
        """

        # TODO: Input validation

        if state_pred is not None:
            self.state_pred = state_pred

        x_pred = self.state_pred.mean
        P_pred = self.state_pred.covar
        H = self.meas_model.eval()
        R = self.meas_model.covar()

        self.meas_pred = GaussianState()
        self.meas_pred.mean, self.meas_pred.covar = self._predict_meas(
            x_pred, P_pred, H, R)

        return self.meas_pred

    @staticmethod
    def _predict_state(x, P, F, Q, u=0, B=1, Qu=0):
        """Low-level Kalman Filter state prediction

        Parameters:
        ----------
        x : 1-D array-like of length Ns
            The prior state mean
        P : 2-D array-like of shape (Ns,Ns)
            The prior state covariance
        F : 2-D array-like of shape (Ns,Ns)
            The state transition matrix
        Q : 2-D array-like of shape (Ns,Ns)
            The process noise covariance matrix
        u : 1-D array-like of length Nu, optional
            The control input (the default is 0, which implies no control
            input)
        B : 2-D array-like of shape (Ns,Nu), optional
            The control gain matrix (the default is 1, which implies no
            control gain)
        Qu : 2-D array-like of shape (Ns,Ns), optional
            The control process covariance matrix (the default is 0, which
            implies no control noise)

        Returns
        -------
        x_pred: 1-D array-like of length Ns
            The predicted state mean
        P_Pred: 2-D array-like of shape (Ns,Ns)
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
        x_pred : 1-D array-like of length Ns
            The predicted state mean
        P_pred : 2-D array-like of shape (Ns,Ns)
            The predicted state covariance
        H : 2-D array-like of shape (Nm,Ns)
            The measurement model matrix
        R : 2-D array-like of shape (Nm,Nm)
            The measurement noise covariance matrix

        Returns
        -------
        y_pred: 1-D array-like of length Nm
            The predicted measurement mean
        S: 2-D array-like of shape (Nm,Nm)
            The predicted measurement noise (innovation) covariance matrix
        """

        y_pred = H@x_pred
        S = H@P_pred@H.T + R

        return y_pred, S
