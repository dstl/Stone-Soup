# -*- coding: utf-8 -*-

import numpy as np


from .base import Predictor
from ..types import State, GaussianStatePrediction


class KalmanPredictor(Predictor):
    """KalmanPredictor class

    An implementation of a standard Kalman Filter predictor.

    """

    def predict(self, prior, control_input=None, timestamp=None, **kwargs):
        """Kalman Filter state prediction step

        Parameters
        ----------
        prior : :class:`~.GaussianState`
            The prior state
        control_input : :class:`~.State`, optional
            The control input. It will only have an effect if
            :attr:`control_model` is not `None` (the default is `None`)
        timestamp: :class:`datetime.datetime`, optional
            A timestamp signifying when the prediction is performed \
            (the default is `None`)

        Returns
        -------
        : :class:`~.GaussianStatePrediction`
            The predicted state

        """

        # Compute time_interval
        try:
            time_interval = timestamp - prior.timestamp
        except TypeError as e:
            # TypeError: (timestamp or prior.timestamp) is None
            time_interval = None

        # Transition model parameters
        transition_matrix = self.transition_model.matrix(
            timestamp=timestamp,
            time_interval=time_interval,
            **kwargs)
        transition_noise_covar = self.transition_model.covar(
            timestamp=timestamp,
            time_interval=time_interval,
            **kwargs)

        # Control model parameters
        if self.control_model is None:
            control_matrix = np.zeros(prior.covar.shape)
            contol_noise_covar = np.zeros(prior.covar.shape)
            control_input = State(np.zeros(prior.state_vector.shape))
        else:
            # Extract control matrix
            control_matrix = self.control_model.matrix(
                timestamp=timestamp,
                time_interval=time_interval,
                **kwargs)
            # Extract control noise covariance
            try:
                # covar() is implemented for control_model
                contol_noise_covar = self.control_model.covar(
                    timestamp=timestamp,
                    time_interval=time_interval,
                    **kwargs)
            except AttributeError as e:
                # covar() is NOT implemented for control_model
                contol_noise_covar = np.zeros(self.control_model.ndim_ctrl)
            if control_input is None:
                control_input = np.zeros((self.control_model.ndim_ctrl, 1))

        # Perform prediction
        prediction_mean, prediction_covar = self.predict_lowlevel(
            prior.mean, prior.covar, transition_matrix,
            transition_noise_covar, control_input.state_vector,
            control_matrix, contol_noise_covar)

        return GaussianStatePrediction(prediction_mean,
                                       prediction_covar,
                                       timestamp)

    @staticmethod
    def predict_lowlevel(x, P, F, Q, u, B, Qu):
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
        : :class:`numpy.ndarray` of shape (Ns,1)
            The predicted state mean
        : :class:`numpy.ndarray` of shape (Ns,Ns)
            The predicted state covariance
        """

        x_pred = F@x + B@u
        P_pred = F@P@F.T + Q + B@Qu@B.T

        return x_pred, P_pred


class ExtendedKalmanPredictor(KalmanPredictor):
    """ExtendedKalmanPredictor class

    An implementation of an Extended Kalman Filter predictor"""

    def predict(self, prior, control_input=None, timestamp=None, **kwargs):
        """ Extended Kalman Filter state prediction step

        Parameters
        ----------
        prior : :class:`~.GaussianState`
            The prior state
        control_input : :class:`~.State`, optional
            The control input. It will only have an effect if
            :attr:`control_model` is not `None` (the default is `None`)
        timestamp: :class:`datetime.datetime`, optional
            A timestamp signifying when the prediction is performed \
            (the default is `None`)

        Returns
        -------
        : :class:`~.GaussianStatePrediction`
            The predicted state
        """

        # Compute time_interval
        try:
            time_interval = timestamp - prior.timestamp
        except TypeError as e:
            # TypeError: (timestamp or prior.timestamp) is None
            time_interval = None

        # Transition model parameters
        try:
            # Attempt to extract matrix from a LinearModel
            transition_matrix = self.transition_model.matrix(
                timestamp=timestamp,
                time_interval=time_interval,
                **kwargs)
        except AttributeError:
            # Else read jacobian from a NonLinearModel
            transition_matrix = self.transition_model.jacobian(
                state_vec=prior.state_vector,
                timestamp=timestamp,
                time_interval=time_interval,
                **kwargs)

        transition_noise_covar = self.transition_model.covar(
            timestamp=timestamp,
            time_interval=time_interval,
            **kwargs)

        # Control model parameters
        if self.control_model is None:
            control_matrix = np.zeros(prior.covar.shape)
            contol_noise_covar = np.zeros(prior.covar.shape)
            control_input = State(np.zeros(prior.state_vector.shape))
        else:
            # Extract control matrix
            try:
                # Attempt to extract matrix from a LinearModel
                control_matrix = self.control_model.matrix(
                    timestamp=timestamp,
                    time_interval=time_interval,
                    **kwargs)
            except AttributeError:
                # Else read jacobian from a NonLinearModel
                control_matrix = self.control_model.jacobian(
                    timestamp=timestamp,
                    time_interval=time_interval,
                    **kwargs)
            # Extract control noise covariance
            try:
                # covar() is implemented for control_model
                contol_noise_covar = self.control_model.covar(
                    timestamp=timestamp,
                    time_interval=time_interval,
                    **kwargs)
            except AttributeError as e:
                # covar() is NOT implemented for control_model
                contol_noise_covar = np.zeros((self.control_model.ndim_ctrl,
                                               self.control_model.ndim_ctrl))
            if control_input is None:
                control_input = np.zeros((self.control_model.ndim_ctrl, 1))

        # Perform state prediction
        prediction_mean, prediction_covar = super().predict_lowlevel(
            prior.mean, prior.covar, transition_matrix,
            transition_noise_covar, control_input.state_vector,
            control_matrix, contol_noise_covar)

        return GaussianStatePrediction(prediction_mean,
                                       prediction_covar,
                                       timestamp)
