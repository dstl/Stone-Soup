# -*- coding: utf-8 -*-

import numpy as np
import datetime as datetime

from ..base import Property
from .base import Predictor
from ..types import GaussianStatePrediction
from ..models import TransitionModel, ControlModel
from ..models.control.linear import LinearControlModel
from ..functions import gauss2sigma, unscented_transform



class AbstractKalmanPredictor(Predictor):
    """
    A predictor class which follows the family of Kalman predictors.

    Generally:
        :math:`\hat{x}_k = f_k(x_{k-1}) + b_k(u_k) + \nu_k`

    where :math:`x_{k-1}` is the prior state, :math:`f_k(\dot)` is the transition function, :math:`u_k` the control
    vector, :math:`b_k(\dot)` the control input and :math:`\nu_k` the noise.

    """

    transition_model = Property(TransitionModel, default=None, doc="The transition model to be used")
    control_model = Property(ControlModel, default=None, doc="The control model to be used")

    def transition_matrix(self, prior=None, **kwargs):
        pass

    def transition_function(self, prior, **kwargs):
        """
        This is :math:`f_k(x_{k-1})`, the transition function, non-linear in general.
        :param prior:
        :param kwargs: might include time stamp or time interval
        :return:
        """
        return self.transition_model.function(prior.state_vector, **kwargs)

    def control_matrix(self, **kwargs):
        pass

    def control_function(self, **kwargs):
        return self.control_model.control_input()

    def _predict_over_interval(self, prior, timestamp):
        """
        Private function to get the prediction interval (or None)

        :param prior: the prior state
        :param timestamp: (current) timestamp
        :return: time interval to predict over
        """

        # Deal with undefined timestamps
        if (timestamp is None) or (prior.timestamp is None):
            predict_over_interval = None
        else:
            predict_over_interval = timestamp - prior.timestamp

        return predict_over_interval

    def _control_model(self, prior):
        """
        Private. If the control model doesn't exist, create it
        :param prior:
        :return:
        """
        # Deal with undefined control model
        if self.control_model is None:
            """Make a 0-effect control input"""
            control_model = LinearControlModel(prior.ndim, [], np.zeros(prior.state_vector.shape),
                                                     np.zeros(prior.covar.shape), np.zeros(prior.covar.shape))
        else:
            control_model = self.control_model

        return control_model

    def predict(self, prior, timestamp=None, **kwargs):
        """
        The predict step,

        :param prior: :math:`x_{t-1}`
        :param timestamp: :math:`t`
        :param kwargs:
        :return: :math:`\hat{x}_k`, the predicted state
        """

        # Get the prediction interval
        predict_over_interval = self._predict_over_interval(prior, timestamp)

        control_model = self._control_model(prior)

        # Prediction of the mean
        x_pred = self.transition_function(prior, time_interval=predict_over_interval) + \
                 control_model.control_input()

        # As this is Kalman-like, the control model must be capable of returning a control matrix (B)
        p_pred = self.transition_matrix(prior=prior, time_interval=predict_over_interval) @ prior.covar @ \
                 self.transition_matrix(prior=prior, time_interval=predict_over_interval).T + \
                 self.transition_model.covar(time_interval=predict_over_interval) + \
                 control_model.matrix() @ control_model.control_noise @ control_model.matrix().T

        return GaussianStatePrediction(x_pred, p_pred, timestamp=timestamp)


class KalmanPredictor(AbstractKalmanPredictor):
    """
    KalmanPredictor class

    An implementation of a standard Kalman Filter predictor. Here:

    :math:`f_k(x_{k-1}) = F_k x_{k-1}`

    """

    # TODO specify that transition and control models _must_ be linear

    def transition_matrix(self, **kwargs):
        """
        Return the transition matrix

        :param kwargs:
        :return: the transition matrix
        """
        return self.transition_model.matrix(**kwargs)

    def transition_function(self, prior, **kwargs):
        """
        Applies the linear transition function, returns the predicted state, :math:`\hat{x}_{k|k-1}`
        :param prior: the prior state, :math:`x_{k-1}`
        :param kwargs:
        :return: the predicted state, :math:`\hat{x}_{k|k-1}`
        """
        return self.transition_model.matrix(**kwargs) @ prior.state_vector


class ExtendedKalmanPredictor(AbstractKalmanPredictor):
    """
    ExtendedKalmanPredictor class

    An implementation of the Extended Kalman Filter predictor. Here the transition and control functions may be
    non-linear, their transition and control matrices are approximated via Jacobian matrices.

    """

    # TODO specify that transition and control models must be 'linearisable' via Jacobians
    def transition_matrix(self, prior, **kwargs):
        """

        :param prior: the prior state, :math:`x_{k-1}`
        :param kwargs:
        :return: the predicted state, :math:`\hat{x}_{k|k-1}`
        """
        if hasattr(self.transition_model, 'matrix'):
            return self.transition_model.matrix(**kwargs)
        else:
            return self.transition_model.jacobian(prior.state_vector, **kwargs)

    # TODO work out how these incorporate time intervals
    # TODO there may also be compelling reason to keep these linear
    def control_matrix(self):
        if hasattr(self.control_model, 'matrix'):
            return self.control_model.matrix()
        else:
            return self.control_model.jacobian(self.control_model.control_vector)


class UnscentedKalmanPredictor(AbstractKalmanPredictor):
    """
    UnscentedKalmanFilter class

    The predict is accomplished by calculating the sigma points from the Gaussian mean and covariance, then putting
    these through the (in general non-linear) transition function, then reconstructing the Gaussian.

    """
    alpha = Property(float, default=0.5, doc="Primary sigma point spread scaling parameter. Typically 0.5.")
    beta = Property(float, default=2, doc="Used to incorporate prior knowledge of the distribution.\
                            If the true distribution is Gaussian, the value of 2 is optimal.")
    kappa = Property(float, default=0, doc="Secondary spread scaling parameter (default is calculated as 3-Ns)" )

    _time_interval = Property(datetime.timedelta, default=None, doc="Hidden variable where time interval is optionally "
                                                                    "stored")

    def transition_and_control_function(self, prior_state_vector, **kwargs):
        """
        Returns the result of applying the transition and control functions for the unscented transform

        :param prior_state_vector: Prior state vector (Requires some unsatisfactory fudges because unscented transform
                                    operates on the state vector rather than state.
        :param kwargs:
        :return:
        """

        predict_over_interval = self._time_interval
        return self.transition_model.function(prior_state_vector, time_interval=predict_over_interval, **kwargs) + \
               self.control_model.control_input()

    def predict(self, prior, timestamp=None, **kwargs):
        """
        The unscented version of the predict step

        :param prior: Prior state
        :param timestamp: time to transit to
        :param kwargs:
        :return:
        """

        # Get the prediction interval
        predict_over_interval = self._predict_over_interval(prior, timestamp )
        self._time_interval = predict_over_interval  # This ensures that function passed to unscented transform has time
                                                    # interval.

        # The control model
        control_model = self._control_model(prior)
        self.control_model = control_model  # This ensures that function passed to unscented transform works

        # The covariance on the transition model + the control model
        # TODO Note that I'm not sure you can actually do this with the covariances, i.e. sum them before calculating
        # TODO the sigma points and then just sticking them into the unscented transform, and I haven't checked the
        # TODO statistics.
        total_noise_covar = self.transition_model.covar(timestamp=timestamp, time_interval=predict_over_interval,
                                                        **kwargs) + control_model.control_noise

        # Get the sigma points from the prior mean and covariance.
        sigma_points, mean_weights, covar_weights = gauss2sigma(prior.state_vector, prior.covar,
                                                                self.alpha, self.beta, self.kappa)

        # Put these through the unscented transform, together with the total covariance to get the parameters of the
        # Gaussian
        x_pred, p_pred, _, _, _, _ = unscented_transform(sigma_points, mean_weights, covar_weights,
                                                         self.transition_and_control_function,
                                                         covar_noise=total_noise_covar)

        # and return a Gaussian state based on these parameters
        return GaussianStatePrediction(x_pred, p_pred, timestamp=timestamp)