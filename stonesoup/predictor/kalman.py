# -*- coding: utf-8 -*-

import numpy as np

from ..base import Property
from .base import Predictor
from ..types import GaussianStatePrediction
from ..models import TransitionModel, ControlModel
from ..models.control.linear import LinearControlModel


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

    def transition_matrix(self, time_interval):
        pass

    def transition_function(self, prior, timestamp):
        pass

    def control_matrix(self):
        pass

    def control_function(self):
        pass

    def predict(self, prior, timestamp=None, **kwargs):
        """

        :param prior: :math:`x_{t-1}`
        :param timestamp: :math:`t`
        :param kwargs:
        :return: :math:`\hat{x}_k`, the predicted state
        """

        # WARNING - this won't work with an undefined timestamp

        if self.control_model is None:
            """Make a 0-effect control input"""
            self.control_model = LinearControlModel(prior.ndim, [], np.zeros(prior.state_vector.shape),
                                                    np.zeros(prior.covar.shape),
                                                    np.zeros(prior.covar.shape))

        # TODO time interval not currently correctly handled - specified externally.
        x_pred = self.transition_function(prior.state_vector) + \
                 self.control_function(self.control_model.control_vector)

        # As this is Kalman-like, the control model must be capable of returning a control matrix (B)
        P_pred = self.transition_matrix() @ prior.covariance @ \
                 self.transition_matrix().T + \
                 self.transition_model.covar() + \
                 self.control_matrix() @ self.control_model.control_noise @ self.control_matrix().T

        return GaussianStatePrediction(x_pred, P_pred)


class KalmanPredictor(AbstractKalmanPredictor):

    """KalmanPredictor class

    An implementation of a standard Kalman Filter predictor.

    """

    # TODO specify that transition and control models must be linear

    def transition_matrix(self):
        return self.transition_model.matrix()

    def transition_function(self, prior, time_interval):
        return self.transition_model.matrix() @ prior.state_vector

    def control_matrix(self):
        return self.control_model.control_matrix

    def control_function(self):
        return self.control_input()


class ExtendedKalmanPredictor(AbstractKalmanPredictor):
    """ExtendedKalmanPredictor class

    An implementation of the Extended Kalman Filter predictor.

    """

    # TODO specify that transition and control models must be 'linearisable' via Jacobians

    def transition_matrix(self, prior):
        return self.transition_model.jacobian(prior.state_vector)

    def transition_function(self, prior):
        return self.transition_model.transition(prior.state_vector)

    # TODO work out how these incorporate time intervals
    # TODO there may also be compelling reason to keep these linear
    def control_matrix(self):
        return self.control_model.jacobian(self.control_model.control_vector)

    def control_function(self):
        return self.control_input()

