# -*- coding: utf-8 -*-

import numpy as np

from ..base import Property,
from .base import Updater, MeasurementModel
from ..types import GaussianMeasurementPrediction, GaussianStateUpdate
from ..models import LinearGaussian
from ..models.measurement import MeasurementModel, NonLinearModel


class AbstractKalmanUpdater(Updater):
    """
    An abstract class which embodies much of the functional infrastructure inherent in Kalman-type updaters;
    will allow daughter classes merely to specify the measurement model h(x)

    Note that this isn't an abstract class as such. It will execute the functions
    """

    # TODO at present this will throw an error if a measurement model is not specified. Either remove default=None
    # TODO or specify some sort of behaviour if default is none.
    measurement_model = Property(MeasurementModel, default=None, doc="The measurement model to be used")

    def measurement_matrix(self, predicted_state=None, **kwargs):
        pass

    def predict_measurement(self, predicted_state, **kwargs):
        """
        :param predicted_state: The predicted state :math:`\hat{\mathbf{x}}_{k|k-1}`
        :param kwargs:
        :return: A Gaussian measurement prediction, :math:`\hat{\mathbf{z}}_{k}`
        """

        pred_meas = self.measurement_model.function(predicted_state.state_vector, noise=[0])

        hh = self.measurement_matrix(predicted_state)

        innov_cov = hh @ predicted_state.covar @ hh.T + self.measurement_model.noise_covar
        meas_cross_cov = predicted_state.covar @ hh.T

        return GaussianMeasurementPrediction(pred_meas, innov_cov, predicted_state.timestamp, meas_cross_cov)

    def update(self, hypothesis, **kwargs):
        """
        The Kalman-type update method

        :param hypothesis: the predicted measurement-measurement association hypothesis
        :param kwargs:
        :return: Posterior state, :math:`\mathbf{x}_{k|k}`
        """

        predicted_state = hypothesis.prediction # Get the predicted state out of the hypothesis

        gmp = self.predict_measurement(predicted_state, **kwargs) # Get the measurement prediction
        pred_meas = gmp.state_vector
        innov_cov = gmp.covar
        m_cross_cov = gmp.cross_covar

        # Complete the calculation of the posterior
        kalman_gain = m_cross_cov @ np.linalg.inv(innov_cov) # This isn't optimised
        posterior_mean = predicted_state.state_vector + kalman_gain @ (hypothesis.measurement.state_vector - pred_meas)
        posterior_covariance = predicted_state.covar - kalman_gain @ innov_cov @ kalman_gain.T

        #posterior_state_covariance = (P_post + P_post.T) / 2 # !!! kludge

        return GaussianStateUpdate(posterior_mean, posterior_covariance, hypothesis, hypothesis.measurement.timestamp)


class KalmanUpdater(AbstractKalmanUpdater):
    """
    Kalman Updater

    Perform measurement update step as in the standard Kalman Filter. Assumes the measurement matrix function of the
    measurement_model returns a matrix (H).

    """

    measurement_model = Property(LinearGaussian, default=None, doc="A linear Gaussian measurement model")

    def measurement_matrix(self, predicted_state):
        """
        This is straightforward Kalman so just get the Matrix from the measurement model
        :return: the measurement matrix, :math:`H_k`
        """
        return self.measurement_model.matrix()


class ExtendedKalmanUpdater(AbstractKalmanUpdater):
    """
    The EKF version of the Kalman Updater

    The measurement model must be non-linear and return the linearisation of h() via the jacobian matrix H

    """

    measurement_model = Property(MeasurementModel, default=None, doc="A non-linear differentiable measurement model")

    def measurement_matrix(self, predicted_state):
        """

        :param predicted_state: the predicted state, :math:`\hat{\mathbf{x}}_{k|k-1}`
        :return: the measurement matrix, :math:`H_k`
        """
        if hasattr( self.measurement_model, 'matrix' ):
            return self.measurement_model.matrix()
        else:
            return self.measurement_model.jacobian(predicted_state.state_vector)


class UnscentedKalmanUpdater(AbstractKalmanUpdater):
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

    def predict_measurement(self, predicted_state, **kwargs):

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
            hypothesis = SingleHypothesis(hypothesis.prediction,
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
        alpha : float
            Spread of the sigma points.
        beta : float
            Used to incorporate prior knowledge of the distribution
            2 is optimal is the state is normally distributed.
        kappa : float
            Secondary spread scaling parameter

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
        alpha : float
            Spread of the sigma points.
        beta : float
            Used to incorporate prior knowledge of the distribution
            2 is optimal is the state is normally distributed.
        kappa : float
            Secondary spread scaling parameter

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
