# -*- coding: utf-8 -*-

import numpy as np


from .base import Predictor
from ..types.state import GaussianState, State


class KalmanPredictor(Predictor):
    """KalmanPredictor class

    An implementation of a standard Kalman Filter predictor.

    """

    def __init__(self, transition_model, measurement_model=None,
                 control_model=None, *args, **kwargs):
        """Constructor method"""

        super().__init__(transition_model, measurement_model,
                         control_model, *args, **kwargs)

    def predict(self, state, control_input=None, timestamp=None, **kwargs):
        """Kalman Filter full prediction step

        Parameters
        ----------
        state : :class:`stonesoup.types.state.GaussianState`
            A prior state object
        control_input : :class:`stonesoup.types.state.State`, optional
            The control input. It will only have an effect if
            :attr:`control_model` is not `None` (the default is `None`)
        timestamp: :class:`datetime.datetime`, optional
            A timestamp signifying when the prediction is performed \
            (the default is `None`)

        Returns
        -------
        :class:`stonesoup.types.state.GaussianState`
            The predicted state
        :class:`stonesoup.types.state.GaussianState`
            The measurement prediction
        :class:`numpy.ndarray` of shape (Nm,Nm)
            The calculated state-to-measurement cross covariance
        """

        state_pred = self.predict_state(state, control_input,
                                        timestamp, **kwargs)
        meas_pred, cross_covar = self.predict_measurement(state_pred,
                                                          **kwargs)

        return state_pred, meas_pred, cross_covar

    def predict_state(self, state, control_input=None,
                      timestamp=None, **kwargs):
        """Kalman Filter state prediction step

        Parameters
        ----------
        state : :class:`stonesoup.types.state.GaussianState`
            The prior state
        control_input : :class:`stonesoup.types.state.State`, optional
            The control input. It will only have an effect if
            :attr:`control_model` is not `None` (the default is `None`)
        timestamp: :class:`datetime.datetime`, optional
            A timestamp signifying when the prediction is performed \
            (the default is `None`)

        Returns
        -------
        :class:`stonesoup.types.state.GaussianState`
            The predicted state

        """

        # Compute time_interval
        try:
            time_interval = timestamp - state.timestamp
        except TypeError as e:
            # TypeError: (timestamp or state.timestamp) is None
            time_interval = None

        # Transition model parameters
        transition_matrix = self.transition_model.matrix(
            timestamp=timestamp,
            time_interval=time_interval)
        transition_noise_covar = self.transition_model.covar(
            timestamp=timestamp,
            time_interval=time_interval)

        # Control model parameters
        if self.control_model is None:
            control_matrix = np.array(np.zeros((2, 2)))
            contol_noise_covar = np.array(np.zeros((2, 2)))
            control_input = State(np.array(np.zeros((2, 1))))
        else:
            # Extract control matrix
            control_matrix = self.control_model.matrix(
                timestamp=timestamp,
                time_interval=time_interval)
            # Extract control noise covariance
            try:
                # covar() is implemented for control_model
                contol_noise_covar = self.control_model.covar(
                    timestamp=timestamp,
                    time_interval=time_interval)
            except AttributeError as e:
                # covar() is NOT implemented for control_model
                contol_noise_covar = np.zeros(self.control_model.ndim_ctrl)
            if control_input is None:
                control_input = np.zeros((self.control_model.ndim_ctrl, 1))

        # Perform state prediction
        state_pred_mean, state_pred_covar = self._predict_state(
            state.mean, state.covar, transition_matrix,
            transition_noise_covar, control_input.state_vector,
            control_matrix, contol_noise_covar)

        return GaussianState(state_pred_mean, state_pred_covar, timestamp)

    def predict_measurement(self, state, **kwargs):
        """Kalman Filter measurement prediction step

        Parameters
        ----------
        state : :class:`stonesoup.types.state.GaussianState`
            A predicted state object

        Returns
        -------
        :class:`stonesoup.types.state.GaussianState`
            The measurement prediction
        :class:`numpy.ndarray` of shape (Nm,Nm)
            The predicted measurement noise (innovation) covariance matrix
        """

        # Measurement model parameters
        measurement_matrix = self.measurement_model.matrix(**kwargs)
        measurement_noise_covar = self.measurement_model.covar(**kwargs)

        meas_pred_mean, meas_pred_covar, cross_covar = \
            self._predict_meas(state.mean, state.covar,
                               measurement_matrix, measurement_noise_covar)

        meas_pred = GaussianState(meas_pred_mean,
                                  meas_pred_covar,
                                  state.timestamp)
        return meas_pred, cross_covar

    @staticmethod
    def _predict_state(x, P, F, Q, u, B, Qu):
        """Low-level Kalman Filter state prediction

        Parameters
        ----------
        x : :class:`numpy.ndarray` of shape (Ns,1)
            The prior state mean
        P : :class:`numpy.ndarray` of shape (Ns,Ns)
            The prior state covariance
        F : :class:`numpy.ndarray` of shape (Ns,Ns)
            The state transition matrix
        Q : :class:`numpy.ndarray` of shape (Ns,Ns)
            The process noise covariance matrix
        u : :class:`numpy.ndarray` of shape (Nu,1)
            The control input
        B : :class:`numpy.ndarray` of shape (Ns,Nu)
            The control gain matrix
        Qu : :class:`numpy.ndarray` of shape (Ns,Ns)
            The control process covariance matrix

        Returns
        -------
        :class:`numpy.ndarray` of shape (Ns,1)
            The predicted state mean
        :class:`numpy.ndarray` of shape (Ns,Ns)
            The predicted state covariance
        """

        x_pred = F@x + B@u
        P_pred = F@P@F.T + Q + B@Qu@B.T

        return x_pred, P_pred

    @staticmethod
    def _predict_meas(x_pred, P_pred, H, R):
        """Low-level Kalman Filter measurement prediction

        Parameters
        ----------
        x_pred : :class:`numpy.ndarray` of shape (Ns,1)
            The predicted state mean
        P_pred : :class:`numpy.ndarray` of shape (Ns,Ns)
            The predicted state covariance
        H : :class:`numpy.ndarray` of shape (Nm,Ns)
            The measurement model matrix
        R : :class:`numpy.ndarray` of shape (Nm,Nm)
            The measurement noise covariance matrix

        Returns
        -------
        y_pred: :class:`numpy.ndarray` of shape (Nm,1)
            The predicted measurement mean
        S: :class:`numpy.ndarray` of shape (Nm,Nm)
            The predicted measurement noise (innovation) covariance matrix
        Pxy: :class:`numpy.ndarray` of shape (Nm,Nm)
            The calculated state-to-measurement cross-covariance
        """

        y_pred = H@x_pred
        S = H@P_pred@H.T + R
        Pxy = P_pred@H.T

        return y_pred, S, Pxy


class ExtendedKalmanPredictor(KalmanPredictor):
    """ExtendedKalmanPredictor class

    An implementation of an Extended Kalman Filter predictor"""

    def __init__(self, transition_model, measurement_model=None,
                 control_model=None, *args, **kwargs):
        """Constructor method"""

        super().__init__(transition_model, measurement_model,
                         control_model, *args, **kwargs)

    def predict(self, state, control_input=None, timestamp=None, **kwargs):
        """Extended Kalman Filter full prediction step

        Parameters
        ----------
        state : :class:`stonesoup.types.state.GaussianState`
            A prior state object
        control_input : :class:`stonesoup.types.state.State`, optional
            The control input. It will only have an effect if
            :attr:`control_model` is not `None` (the default is `None`)
        timestamp: :class:`datetime.datetime`, optional
            A timestamp signifying when the prediction is performed \
            (the default is `None`)

        Returns
        -------
        :class:`stonesoup.types.state.GaussianState`
            The predicted state
        :class:`stonesoup.types.state.GaussianState`
            The measurement prediction
        :class:`numpy.ndarray` of shape (Nm,Nm)
            The calculated state-to-measurement cross covariance
        """

        return super().predict(state=state,
                               control_input=control_input,
                               timestamp=timestamp,
                               **kwargs)

    def predict_state(self, state, control_input=None,
                      timestamp=None, **kwargs):
        """ Extended Kalman Filter state prediction step

        Parameters
        ----------
        state : :class:`stonesoup.types.state.GaussianState`
            The prior state
        control_input : :class:`stonesoup.types.state.State`, optional
            The control input. It will only have an effect if
            :attr:`control_model` is not `None` (the default is `None`)
        timestamp: :class:`datetime.datetime`, optional
            A timestamp signifying when the prediction is performed \
            (the default is `None`)

        Returns
        -------
        :class:`stonesoup.types.state.GaussianState`
            The predicted state
        """

        # Compute time_interval
        try:
            time_interval = timestamp - state.timestamp
        except TypeError as e:
            # TypeError: (timestamp or state.timestamp) is None
            time_interval = None

        # Transition model parameters
        try:
            # Attempt to extract matrix from a LinearModel
            transition_matrix = self.transition_model.matrix(
                timestamp=timestamp,
                time_interval=time_interval)
        except AttributeError:
            # Else read jacobian from a NonLinearModel
            transition_matrix = self.transition_model.jacobian(
                timestamp=timestamp,
                time_interval=time_interval)

        transition_noise_covar = self.transition_model.covar(
            timestamp=timestamp,
            time_interval=time_interval)

        # Control model parameters
        if self.control_model is None:
            control_matrix = np.array(np.zeros((2, 2)))
            contol_noise_covar = np.array(np.zeros((2, 2)))
            control_input = State(np.array(np.zeros((2, 1))))
        else:
            # Extract control matrix
            try:
                # Attempt to extract matrix from a LinearModel
                control_matrix = self.control_model.matrix(
                    timestamp=timestamp,
                    time_interval=time_interval)
            except AttributeError:
                # Else read jacobian from a NonLinearModel
                control_matrix = self.control_model.jacobian(
                    timestamp=timestamp,
                    time_interval=time_interval)
            # Extract control noise covariance
            try:
                # covar() is implemented for control_model
                contol_noise_covar = self.control_model.covar(
                    timestamp=timestamp,
                    time_interval=time_interval)
            except AttributeError as e:
                # covar() is NOT implemented for control_model
                contol_noise_covar = np.zeros((self.control_model.ndim_ctrl,
                                               self.control_model.ndim_ctrl))
            if control_input is None:
                control_input = np.zeros((self.control_model.ndim_ctrl, 1))

        # Perform state prediction
        state_pred_mean, state_pred_covar = super()._predict_state(
            state.mean, state.covar, transition_matrix,
            transition_noise_covar, control_input.state_vector,
            control_matrix, contol_noise_covar)

        return GaussianState(state_pred_mean, state_pred_covar, timestamp)

    def predict_measurement(self, state, **kwargs):
        """Extended Kalman Filter measurement prediction step

        Parameters
        ----------
        state : :class:`stonesoup.types.state.GaussianState`
            A predicted state object

        Returns
        -------
        :class:`stonesoup.types.state.GaussianState`
            The measurement prediction
        :class:`numpy.ndarray` of shape (Nm,Nm)
            The predicted measurement noise (innovation) covariance matrix
        """

        # Measurement model parameters
        try:
            # Attempt to extract matrix from a LinearModel
            measurement_matrix = self.measurement_model.matrix(**kwargs)
        except AttributeError:
            # Else read jacobian from a NonLinearModel
            measurement_matrix = self.measurement_model.jacobian(**kwargs)

        measurement_noise_covar = self.measurement_model.covar(**kwargs)

        meas_pred_mean, meas_pred_covar, cross_covar = \
            super()._predict_meas(state.mean, state.covar,
                                  measurement_matrix, measurement_noise_covar)

        meas_pred = GaussianState(meas_pred_mean,
                                  meas_pred_covar,
                                  state.timestamp)
        return meas_pred, cross_covar
