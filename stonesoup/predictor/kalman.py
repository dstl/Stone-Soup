# -*- coding: utf-8 -*-

import numpy as np
import datetime as datetime

from ..base import Property
from .base import Predictor
from ..types.prediction import GaussianStatePrediction
from ..models.transition import TransitionModel
from ..models.transition.linear import LinearGaussianTransitionModel
from ..models.control import ControlModel
from ..models.control.linear import LinearControlModel
from ..functions import gauss2sigma, unscented_transform


class KalmanPredictor(Predictor):
    r"""
    A predictor class which follows the family of Kalman predictors.

    Generally,

    .. math::

        \mathbf{x}_{k|k-1} = f_k(\mathbf{x}_{k-1}) + b_k(\mathbf{u}_k) + \mathbf{\nu}_k

    where :math:`\mathbf{x}_{k-1}` is the prior state, :math:`f_k(\mathbf{x}_{k-1})` is the transition function,
    :math:`\mathbf{u}_k` the control vector, :math:`b_k(\mathbf{u}_k)` the control input and
    :math:`\mathbf{\nu}_k` the noise.

    This class also serves as the (specific) Kalman Filter :class:`~.Predictor` Class. Here
    :math:`f_k(\mathbf{x}_{k-1}) = F_k \mathbf{x}_{k-1}` and :math:`\mathbf{\nu}_k \sim \mathcal{N}(0,Q_k)`
    """

    # In the Kalman filter transition and control models must be linear
    transition_model = Property(LinearGaussianTransitionModel, doc="The transition model to be used. Functions will "
                                                                   "throw a :class:`~.ValueError` if not specified. ")
    control_model = Property(LinearControlModel, default=None, doc="The control model to be used. In the event that "
                                                                   "this is undefined the predictor will create a "
                                                                   "zero-effect linear :class:`~.ControlModel`." )

    def __init__(self, transition_model, control_model=None, *args, **kwargs):
        """
        Explicitly initialise the models and check for existence of transition model, and throw an error if not
        specified.
        """
        super().__init__(transition_model, control_model=None, *args, **kwargs)

        # Chuck an error if there's no transition model
        if self.transition_model is None:
            raise ValueError("A Predictor requires a state transition model")

        # If no control model insert a linear zero-effect one
        ndims = self.transition_model.ndim_state
        if self.control_model is None:
            """Make a 0-effect control input"""
            self.control_model = LinearControlModel(ndims, [], np.zeros([ndims, 1]), np.zeros([ndims, ndims]),
                                                    np.zeros([ndims, ndims]))

    def transition_matrix(self, **kwargs):
        """
        Return the transition matrix

        :param kwargs: these are passed to :class:`~.LinearGaussianTransitionModel`. :attr:`.matrix()`
        :return: the transition matrix, :math:`F_k`
        """
        return self.transition_model.matrix(**kwargs)

    def transition_function(self, prior, **kwargs):
        """
        Applies the linear transition function, returns the predicted state,
        :math:`\mathbf{x}_{k|k-1}`

        :param prior: the prior state, :math:`\mathbf{x}_{k-1}`
        :param kwargs: these are passed to :class:`~.LinearGaussianTransitionModel`. :attr:`.matrix()`
        :return: the predicted state, :math:`\mathbf{x}_{k|k-1}`
        """
        return self.transition_model.matrix(**kwargs) @ prior.state_vector

    @property
    def control_matrix(self):
        """
        Convenience function which returns the control matrix

        :return: control matrix, :math:`B_k`
        """
        return self.control_model.matrix()

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

    def predict(self, prior, timestamp=None, **kwargs):
        """
        The predict step,

        :param prior: :math:`\mathbf{x}_{k-1}`
        :param timestamp: :math:`k`
        :param kwargs: these are passed, via :class:`~.KalmanFilter`. :attr:`transition_function()` to
                       :class:`~.LinearGaussianTransitionModel`. :attr:`.matrix()`

        :return: :math:`\mathbf{x}_{k|k-1}`, the predicted state and the predicted state covariance :math:`P_{k|k-1}`
        """

        # Get the prediction interval
        predict_over_interval = self._predict_over_interval(prior, timestamp)

        # Prediction of the mean
        x_pred = self.transition_function(prior, time_interval=predict_over_interval, **kwargs) + \
                 self.control_model.control_input()

        # As this is Kalman-like, the control model must be capable of returning a control matrix (B)
        p_pred = self.transition_matrix(prior=prior, time_interval=predict_over_interval, **kwargs) @ prior.covar @ \
                 self.transition_matrix(prior=prior, time_interval=predict_over_interval, **kwargs).T + \
                 self.transition_model.covar(time_interval=predict_over_interval, **kwargs) + \
                 self.control_matrix @ self.control_model.control_noise @ self.control_matrix.T

        return GaussianStatePrediction(x_pred, p_pred, timestamp=timestamp)


class ExtendedKalmanPredictor(KalmanPredictor):
    """
    ExtendedKalmanPredictor class

    An implementation of the Extended Kalman Filter predictor. Here the transition and control functions may be
    non-linear, their transition and control matrices are approximated via Jacobian matrices. To this end the
    transition and control models, if non-linear, must possess return the :attr:`jacobian()` function.

    """

    # In this version the models can be non-linear, but must have access to the :attr:`jacobian()` function
    # TODO Enforce the presence of :attr:`jacobian()`
    transition_model = Property(TransitionModel, doc="The transition model to be used. Functions will "
                                                     "throw a :class:`~.ValueError` if not specified.")
    control_model = Property(ControlModel, default=None, doc="The control model to be used. If "
                                                             "undefined the predictor will create a "
                                                             "zero-effect linear :class:`~.ControlModel`.")

    # TODO Confirm that this inherits the __init__() function

    def transition_matrix(self, prior, **kwargs):
        """

        :param prior: the prior state, :math:`\mathbf{x}_{k-1}`
        :param kwargs: these are passed to :class:`~.TransitionModel`. :attr:`matrix()` if linear (i.e. the
                       :attr:`matrix()` function exists), or :class:`~.TransitionModel`. :attr:`.jacobian()` if not

        :return:  the transition matrix, :math:`F_k` if linear, or its Jacobian if non-linear
        """
        if hasattr(self.transition_model, 'matrix'):
            return self.transition_model.matrix(**kwargs)
        else:
            return self.transition_model.jacobian(prior.state_vector, **kwargs)

    def transition_function(self, prior, **kwargs):
        """
        This is the application of :math:`f_k(\mathbf{x}_{k-1})`, the transition function, non-linear in general

        :param prior: the prior state, :math:`\mathbf{x}_{k-1}`
        :param kwargs: these are passed to :class:`~.ExtendedKalmanFilter`. :attr:`function()`
        :return: the predicted state, :math:`\mathbf{x}_{k|k-1}`
        """
        return self.transition_model.function(prior.state_vector, **kwargs)

    @property
    def control_matrix(self):
        """
        Returns the control model control matrix, `B_k(\mathrm{u}_k)`, or its linear approximation via a
        Jacobian  approximation. The :class:`~.ControlModel` used must therefore be capable of returning a
        :attr:`jacobian()` function.

        :return: The control model matrix, or its linear approximation
        """
        if hasattr(self.control_model, 'matrix'):
            return self.control_model.matrix()
        else:
            return self.control_model.jacobian(self.control_model.control_vector)


class UnscentedKalmanPredictor(KalmanPredictor):
    """
    UnscentedKalmanFilter class

    The predict is accomplished by calculating the sigma points from the Gaussian mean and covariance, then putting
    these through the (in general non-linear) transition function, then reconstructing the Gaussian.

    """
    # The models may be non-linear and non-differentiable
    transition_model = Property(TransitionModel, doc="The transition model to be used. Functions will "
                                                     "throw a :class:`~.ValueError` if not specified.")
    control_model = Property(ControlModel, default=None, doc="The control model to be used. If "
                                                             "undefined the predictor will create a "
                                                             "zero-effect linear :class:~.`ControlModel`.")

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
                                   operates on the state vector rather than state).
        :param kwargs: these are passed to :class:`~.UnscentedKalmanFilter`. :attr:`function()`
        :return: the predicted state at :math:`k`, :math:`\mathbf{x}_{k|k-1}`
        """

        predict_over_interval = self._time_interval
        return self.transition_model.function(prior_state_vector, time_interval=predict_over_interval, **kwargs) + \
               self.control_model.control_input()

    def predict(self, prior, timestamp=None, **kwargs):
        """
        The unscented version of the predict step

        :param prior: Prior state, :math:`\mathbf{x}_{k-1}`
        :param timestamp: time to transit to (:math:`k`)
        :param kwargs: these are passed to :class:`~.TransitionModel`. :attr:`covar()`
        :return: the predicted state :math:`\mathbf{x}_{k|k-1}` and the predicted state covariance :math:`P_{k|k-1}`
        """

        # Get the prediction interval
        predict_over_interval = self._predict_over_interval(prior, timestamp)
        self._time_interval = predict_over_interval  # This ensures that function passed to unscented transform has the
                                                     # correct time interval

        # The covariance on the transition model + the control model
        # TODO Note that I'm not sure you can actually do this with the covariances, i.e. sum them before calculating
        # TODO the sigma points and then just sticking them into the unscented transform, and I haven't checked the
        # TODO statistics.
        total_noise_covar = self.transition_model.covar(timestamp=timestamp, time_interval=predict_over_interval,
                                                        **kwargs) + self.control_model.control_noise

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
