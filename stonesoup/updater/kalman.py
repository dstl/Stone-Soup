# -*- coding: utf-8 -*-

import numpy as np

from .base import Updater
from ..base import Property
from ..types import (GaussianMeasurementPrediction,
                     GaussianStateUpdate, Hypothesis)
from ..functions import gauss2sigma, unscented_transform


class KalmanUpdater(Updater):
    """Simple Kalman Updater

    Perform measurement update step in the standard Kalman Filter.
    """

    def get_measurement_prediction(self, state_prediction,
                                   measurement_model=None, **kwargs):
        """Kalman Filter measurement prediction step

        Parameters
        ----------
        state_prediction : :class:`~.GaussianStatePrediction`
            A predicted state object
        measurement_model: :class:`~.MeasurementModel`, optional
            The measurement model used to generate the measurement prediction.\
            Should be used in cases where the measurement model is dependent\
            on the received measurement.\
            (the default is ``None``, in which case the updater will use the\
            measurement model specified on initialisation)

        Returns
        -------
        : :class:`~.GaussianMeasurementPrediction`
            The measurement prediction
        """

        # Measurement model parameters
        if measurement_model is None:
            measurement_model = self.measurement_model
        measurement_matrix, measurement_noise_covar = \
            self._extract_model_parameters(measurement_model)

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

        # Extract model parameters
        measurement_matrix, measurement_noise_covar = \
            self._extract_model_parameters(self.measurement_model,
                                           hypothesis.measurement)

        # If no measurement prediction is provided with hypothesis
        if hypothesis.measurement_prediction is None:
            # Perform full update step
            posterior_mean, posterior_covar, meas_pred_mean,\
                meas_pred_covar, cross_covar, _ = \
                self.update_lowlevel(
                    hypothesis.prediction.mean,
                    hypothesis.prediction.covar,
                    measurement_matrix,
                    measurement_noise_covar,
                    hypothesis.measurement.state_vector
                )
            # Augment hypothesis with measurement prediction
            hypothesis = Hypothesis(hypothesis.prediction,
                                    hypothesis.measurement,
                                    GaussianMeasurementPrediction(
                                        meas_pred_mean, meas_pred_covar,
                                        hypothesis.prediction.timestamp,
                                        cross_covar)
                                    )
        else:
            # Otherwise, utilise the provided measurement prediction
            posterior_mean, posterior_covar, _ = \
                self.update_on_measurement_prediction(
                    hypothesis.prediction.mean,
                    hypothesis.prediction.covar,
                    hypothesis.measurement.state_vector,
                    hypothesis.measurement_prediction.mean,
                    hypothesis.measurement_prediction.covar,
                    hypothesis.measurement_prediction.cross_covar,
                    measurement_matrix,
                    measurement_noise_covar
                )

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

        x_post, P_post, K = \
            KalmanUpdater.update_on_measurement_prediction(x_pred, P_pred,
                                                           y, y_pred, S,
                                                           Pxy, H, R)

        return x_post, P_post, y_pred, S, Pxy, K

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
    def update_on_measurement_prediction(x_pred, P_pred, y,
                                         y_pred, S, Pxy, H=None, R=None):
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
        H: :class:`numpy.ndarray` of shape (Nm,Nm), optional
            The measurement model matrix. If both `H` and `R` are provided
            then the update will be performed based on the slower, but more
            numerically stable, "Joseph form" update equation:\
            ..:math:
                P_{k|k} = (I-K_kH_k)P_{k|k-1}(I-K_kH_k)^T + K_kR_kK_k^T
            (default is `None`)
        R: :class:`numpy.ndarray` of shape (Nm,Nm), optional
            The measurement model matrix. See information for `H` above.

        Returns
        -------
        : :class:`numpy.ndarray` of shape (Ns,1)
            The computed posterior state mean
        : :class:`numpy.ndarray` of shape (Ns,Ns)
            The computed posterior state covariance
        : :class:`numpy.ndarray` of shape (Ns,Nm)
            The computed Kalman gain
        """

        K = Pxy@np.linalg.pinv(S)

        x_post = x_pred + K@(y-y_pred)

        if(H is not None and R is not None):
            # P = (I-KH)P(I-KH)' + KRK' is more numerically stable
            # and works for non-optimal K vs the equation
            # P = (I-KH)P usually seen in the literature.
            ndim_state = x_pred.shape[0]
            I_KH = np.eye(ndim_state) - K@H
            P_post = I_KH@P_pred@I_KH.T + K@R@K.T
        elif(H is not None):
            ndim_state = x_pred.shape[0]
            P_post = (np.eye(ndim_state) - K@H)@P_pred
        else:
            P_post = P_pred - K@Pxy.T
            P_post = (P_post+P_post.T)/2

        return x_post, P_post, K

    @staticmethod
    def _extract_model_parameters(measurement_model, measurement=None,
                                  **kwargs):
        """Extract measurement model parameters

        Parameters
        ----------
        measurement_model: :class:`~.MeasurementModel`
            A measurement model whose parameters are to be extracted
        measurement : :class:`~.Detection`, optional
            If provided and `measurement.measurement_model` is not `None`,\
            then its parameters will be returned instead\
            (the default is `None`, in which case `self.measurement_model`'s\
            parameters will be returned)

        Returns
        -------
        : :class:`numpy.ndarray` of shape (Nm,Nm)
            The measurement model transformation matrix
        : :class:`numpy.ndarray` of shape (Nm,Nm)
            The measurement model covariance matrix
        """

        if(measurement is not None
           and measurement.measurement_model is not None):
            measurement_matrix = measurement.measurement_model.matrix(**kwargs)
            measurement_noise_covar = measurement.measurement_model.covar(
                **kwargs)
        else:
            measurement_matrix = measurement_model.matrix(**kwargs)
            measurement_noise_covar = measurement_model.covar(**kwargs)

        return measurement_matrix, measurement_noise_covar


class ExtendedKalmanUpdater(KalmanUpdater):
    """Extended Kalman Updater

    Perform measurement update step in the Extended Kalman Filter.
    """

    def get_measurement_prediction(self, state_prediction,
                                   measurement_model=None, **kwargs):
        """Extended Kalman Filter measurement prediction step

        Parameters
        ----------
        state_prediction : :class:`~.GaussianStatePrediction`
            A predicted state object
        measurement_model: :class:`~.MeasurementModel`, optional
            The measurement model used to generate the measurement prediction.\
            Should be used in cases where the measurement model is dependent\
            on the received measurement.\
            (the default is ``None``, in which case the updater will use the\
            measurement model specified on initialisation)

        Returns
        -------
        : :class:`~.GaussianMeasurementPrediction`
            The measurement prediction
        """

        # Measurement model parameters
        if measurement_model is None:
            measurement_model = self.measurement_model
        measurement_matrix, measurement_noise_covar, measurement_function = \
            self._extract_model_parameters(measurement_model)

        meas_pred_mean, meas_pred_covar, cross_covar = \
            self.get_measurement_prediction_lowlevel(state_prediction.mean,
                                                     state_prediction.covar,
                                                     measurement_function,
                                                     measurement_matrix,
                                                     measurement_noise_covar)

        return GaussianMeasurementPrediction(meas_pred_mean, meas_pred_covar,
                                             state_prediction.timestamp,
                                             cross_covar)

        meas_pred_mean, meas_pred_covar, cross_covar = \
            self.get_measurement_prediction_lowlevel(state_prediction.mean,
                                                     state_prediction.covar,
                                                     measurement_function,
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

        # Extract model parameters
        measurement_matrix, measurement_noise_covar, measurement_function = \
            self._extract_model_parameters(self.measurement_model,
                                           hypothesis.prediction.state_vector,
                                           hypothesis.measurement)

        # If no measurement prediction is provided with hypothesis
        if hypothesis.measurement_prediction is None:
            # Perform full update step
            posterior_mean, posterior_covar, meas_pred_mean,\
                meas_pred_covar, cross_covar, _ = \
                self.update_lowlevel(
                    hypothesis.prediction.mean,
                    hypothesis.prediction.covar,
                    measurement_function,
                    measurement_matrix,
                    measurement_noise_covar,
                    hypothesis.measurement.state_vector
                )
            # Augment hypothesis with measurement prediction
            hypothesis = Hypothesis(hypothesis.prediction,
                                    hypothesis.measurement,
                                    GaussianMeasurementPrediction(
                                        meas_pred_mean, meas_pred_covar,
                                        hypothesis.prediction.timestamp,
                                        cross_covar)
                                    )
        else:
            posterior_mean, posterior_covar, _ = \
                self.update_on_measurement_prediction(
                    hypothesis.prediction.mean,
                    hypothesis.prediction.covar,
                    hypothesis.measurement.state_vector,
                    hypothesis.measurement_prediction.mean,
                    hypothesis.measurement_prediction.covar,
                    hypothesis.measurement_prediction.cross_covar,
                    measurement_matrix,
                    measurement_noise_covar
                )

        return GaussianStateUpdate(posterior_mean,
                                   posterior_covar,
                                   hypothesis,
                                   hypothesis.measurement.timestamp)

    @staticmethod
    def update_lowlevel(x_pred, P_pred, h, H, R, y):
        """Low level Extended Kalman Filter update

        Parameters
        ----------
        x_pred: :class:`numpy.ndarray` of shape (Ns,1)
            The predicted state mean
        P_pred: :class:`numpy.ndarray` of shape (Ns,Ns)
            The predicted state covariance
        h : function handle
            The (non-linear) measurement model function
            Must be of the form "y = fun(x)"
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

        y_pred, S, Pxy = \
            ExtendedKalmanUpdater.get_measurement_prediction_lowlevel(x_pred,
                                                                      P_pred,
                                                                      h,
                                                                      H,
                                                                      R)

        x_post, P_post, K = \
            ExtendedKalmanUpdater.update_on_measurement_prediction(x_pred,
                                                                   P_pred,
                                                                   y,
                                                                   y_pred,
                                                                   S,
                                                                   Pxy,
                                                                   H,
                                                                   R)

        return x_post, P_post, y_pred, S, Pxy, K

    @staticmethod
    def get_measurement_prediction_lowlevel(x_pred, P_pred, h, H, R):
        """Low level Extended Kalman Filter measurement prediction

        Parameters
        ----------
        x_pred: :class:`numpy.ndarray` of shape (Ns,1)
            The predicted state mean
        P_pred: :class:`numpy.ndarray` of shape (Ns,Ns)
            The predicted state covariance
        h : function handle
            The (non-linear) measurement model function
            Must be of the form "y = fun(x)"
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
        y_pred = h(x_pred)
        S = H@P_pred@H.T + R
        Pxy = P_pred@H.T

        return y_pred, S, Pxy

    @staticmethod
    def update_on_measurement_prediction(x_pred, P_pred, y,
                                         y_pred, S, Pxy, H=None, R=None):
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

        return KalmanUpdater.update_on_measurement_prediction(x_pred, P_pred,
                                                              y, y_pred, S,
                                                              Pxy, H, R)

    @staticmethod
    def _extract_model_parameters(measurement_model, state_vector=None,
                                  measurement=None, **kwargs):
        """Extract measurement model parameters

        Parameters
        ----------
        measurement_model: :class:`~.MeasurementModel`
            A measurement model whose parameters are to be extracted
        measurement : :class:`~.Detection`, optional
            If provided and `measurement.measurement_model` is not `None`,\
            then its parameters will be returned instead\
            (the default is `None`, in which case `self.measurement_model`'s\
            parameters will be returned)

        Returns
        -------
        : :class:`numpy.ndarray` of shape (Nm,Nm)
            The measurement model transformation matrix
        : :class:`numpy.ndarray` of shape (Nm,Nm)
            The measurement model covariance matrix
        """

        if(measurement is not None
           and measurement.measurement_model is not None):
            return ExtendedKalmanUpdater._extract_model_parameters(
                measurement.measurement_model, state_vector=state_vector)
        else:
            try:
                # Attempt to extract matrix from a LinearModel
                measurement_matrix = measurement_model.matrix(**kwargs)
            except AttributeError:
                # Else read jacobian from a NonLinearModel
                measurement_matrix = \
                    measurement_model.jacobian(state_vector,
                                               **kwargs)

            def measurement_function(x):
                return measurement_model.function(x, noise=0, **kwargs)

            measurement_noise_covar = measurement_model.covar(**kwargs)

        return measurement_matrix, measurement_noise_covar, \
            measurement_function


class UnscentedKalmanUpdater(KalmanUpdater):
    """Unscented Kalman Updater

    Perform measurement update step in the Unscented Kalman Filter.
    """

    alpha = Property(float, default=0.5,
                     doc="Primary sigma point spread scalling parameter.\
                         Typically 1e-3.")
    beta = Property(float, default=2,
                    doc="Used to incorporate prior knowledge of the distribution.\
                        If the true distribution is Gaussian, the value of 2\
                        is optimal.")
    kappa = Property(float, default=0,
                     doc="Secondary spread scaling parameter\
                        (default is calculated as 3-Ns)")

    def get_measurement_prediction(self, state_prediction,
                                   measurement_model=None, **kwargs):
        """Unscented Kalman Filter measurement prediction step

        Parameters
        ----------
        state_prediction : :class:`~.GaussianStatePrediction`
            A predicted state object
        measurement_model: :class:`~.MeasurementModel`, optional
            The measurement model used to generate the measurement prediction.\
            Should be used in cases where the measurement model is dependent\
            on the received measurement.\
            (the default is ``None``, in which case the updater will use the\
            measurement model specified on initialisation)

        Returns
        -------
        : :class:`~.GaussianMeasurementPrediction`
            The measurement prediction
        """

        # Measurement model parameters
        if measurement_model is None:
            measurement_model = self.measurement_model

        measurement_function, measurement_noise_covar = \
            self._extract_model_parameters(measurement_model)

        meas_pred_mean, meas_pred_covar, cross_covar = \
            self.get_measurement_prediction_lowlevel(state_prediction.mean,
                                                     state_prediction.covar,
                                                     measurement_function,
                                                     measurement_noise_covar,
                                                     self.alpha, self.beta,
                                                     self.kappa)

        return GaussianMeasurementPrediction(meas_pred_mean, meas_pred_covar,
                                             state_prediction.timestamp,
                                             cross_covar)

        meas_pred_mean, meas_pred_covar, cross_covar = \
            self.get_measurement_prediction_lowlevel(state_prediction.mean,
                                                     state_prediction.covar,
                                                     measurement_function,
                                                     measurement_noise_covar,
                                                     self.alpha, self.beta,
                                                     self.kappa)
        return GaussianMeasurementPrediction(meas_pred_mean, meas_pred_covar,
                                             state_prediction.timestamp,
                                             cross_covar)

    def update(self, hypothesis, **kwargs):
        """ Unscented Kalman Filter update step

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

        # Extract model parameters
        measurement_function, measurement_noise_covar = \
            self._extract_model_parameters(self.measurement_model,
                                           hypothesis.measurement)

        # If no measurement prediction is provided with hypothesis
        if hypothesis.measurement_prediction is None:
            # Perform full update step
            posterior_mean, posterior_covar, meas_pred_mean,\
                meas_pred_covar, cross_covar, _ = \
                self.update_lowlevel(
                    hypothesis.prediction.mean,
                    hypothesis.prediction.covar,
                    measurement_function,
                    measurement_noise_covar,
                    hypothesis.measurement.state_vector,
                    self.alpha, self.beta, self.kappa
                )
            # Augment hypothesis with measurement prediction
            hypothesis = Hypothesis(hypothesis.prediction,
                                    hypothesis.measurement,
                                    GaussianMeasurementPrediction(
                                        meas_pred_mean, meas_pred_covar,
                                        hypothesis.prediction.timestamp,
                                        cross_covar)
                                    )
        else:
            posterior_mean, posterior_covar, _ =\
                self.update_on_measurement_prediction(
                    hypothesis.prediction.mean,
                    hypothesis.prediction.covar,
                    hypothesis.measurement.state_vector,
                    hypothesis.measurement_prediction.mean,
                    hypothesis.measurement_prediction.covar,
                    hypothesis.measurement_prediction.cross_covar
                )

        return GaussianStateUpdate(posterior_mean,
                                   posterior_covar,
                                   hypothesis,
                                   hypothesis.measurement.timestamp)

    @staticmethod
    def update_lowlevel(x_pred, P_pred, h, R, y, alpha, beta, kappa):
        """Low level Unscented Kalman Filter update

        Parameters
        ----------
        x_pred: :class:`numpy.ndarray` of shape (Ns,1)
            The predicted state mean
        P_pred: :class:`numpy.ndarray` of shape (Ns,Ns)
            The predicted state covariance
        h : function handle
            The (non-linear) measurement model function
            Must be of the form "y = fun(x,w)"
        R : :class:`numpy.ndarray` of shape (Nm,Nm)
            The measurement noise covariance matrix
        y : :class:`numpy.ndarray` of shape (Nm,1)
            The measurement vector
        alpha : float, optional
            Spread of the sigma points. Typically 1e-3.
            (default is 1e-3)
        beta : float, optional
            Used to incorporate prior knowledge of the distribution
            2 is optimal is the state is normally distributed.
            (default is 2)
        kappa : float, optional
            Secondary spread scaling parameter
            (default is calculated as `3-Ns`)

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
            UnscentedKalmanUpdater.get_measurement_prediction_lowlevel(
                x_pred, P_pred, h, R,
                alpha, beta, kappa)

        x_post, P_post, K = \
            UnscentedKalmanUpdater.update_on_measurement_prediction(
                x_pred, P_pred, y, y_pred, S, Pxy)

        return x_post, P_post, y_pred, S, Pxy, K

    @staticmethod
    def get_measurement_prediction_lowlevel(x_pred, P_pred, h, R,
                                            alpha, beta, kappa):
        """Low level Unscented Kalman Filter measurement prediction

        Parameters
        ----------
        x_pred: :class:`numpy.ndarray` of shape (Ns,1)
            The predicted state mean
        P_pred: :class:`numpy.ndarray` of shape (Ns,Ns)
            The predicted state covariance
        h : function handle
            The (non-linear) measurement model function
            Must be of the form "y = fun(x,w)"
        R : :class:`numpy.ndarray` of shape (Nm,Nm)
            The measurement noise covariance matrix
        alpha : float, optional
            Spread of the sigma points. Typically 1e-3.
            (default is 1e-3)
        beta : float, optional
            Used to incorporate prior knowledge of the distribution
            2 is optimal is the state is normally distributed.
            (default is 2)
        kappa : float, optional
            Secondary spread scaling parameter
            (default is calculated as `3-Ns`)

        Returns
        -------
        : :class:`numpy.ndarray` of shape (Nm,1)
            The predicted measurement mean
        : :class:`numpy.ndarray` of shape (Nm,Nm)
            The predicted measurement noise (innovation) covariance matrix
        : :class:`numpy.ndarray` of shape (Ns,Nm), optional
            The state-to-measurement cross covariance
        """

        sigma_points, mean_weights, covar_weights = \
            gauss2sigma(x_pred, P_pred, alpha, beta, kappa)

        y_pred, S, Pxy, _, _, _ = unscented_transform(sigma_points,
                                                      mean_weights,
                                                      covar_weights,
                                                      h, covar_noise=R)

        return y_pred, S, Pxy

    @staticmethod
    def update_on_measurement_prediction(x_pred, P_pred, y,
                                         y_pred, S, Pxy):
        """Low level Unscented Kalman Filter update, based on a measurement\
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

        return KalmanUpdater.update_on_measurement_prediction(x_pred, P_pred,
                                                              y, y_pred, S,
                                                              Pxy)

    @staticmethod
    def _extract_model_parameters(measurement_model, measurement=None,
                                  **kwargs):
        """Extract measurement model parameters

        Parameters
        ----------
        measurement_model: :class:`~.MeasurementModel`
            A measurement model whose parameters are to be extracted
        measurement : :class:`~.Detection`, optional
            If provided and `measurement.measurement_model` is not `None`,\
            then its parameters will be returned instead\
            (the default is `None`, in which case `self.measurement_model`'s\
            parameters will be returned)

        Returns
        -------
        : function handle
            The (non-linear) measurement model function
        : :class:`numpy.ndarray` of shape (Nm,Nm)
            The measurement model covariance matrix
        """

        if(measurement is not None
           and measurement.measurement_model is not None):
            return UnscentedKalmanUpdater._extract_model_parameters(
                measurement.measurement_model)
        else:
            def measurement_function(x, w=0):
                return measurement_model.function(x, w, **kwargs)

            measurement_noise_covar = measurement_model.covar(**kwargs)

        return measurement_function, measurement_noise_covar
