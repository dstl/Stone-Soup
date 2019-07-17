# -*- coding: utf-8 -*-

import numpy as np
import datetime as datetime
from functools import lru_cache

from ..base import Property
from .base import Predictor
from ..types.prediction import GaussianStatePrediction
from ..models.transition import TransitionModel
from ..models.transition.linear import LinearGaussianTransitionModel
from ..models.control import ControlModel
from ..models.control.linear import LinearControlModel
from ..functions import gauss2sigma, unscented_transform


class KalmanPredictor(Predictor):
    r"""A predictor class which forms the basis for the family of Kalman
    predictors. This class also serves as the (specific) Kalman Filter
    :class:`~.Predictor` class. Here

    .. math::

      f_k( \mathbf{x}_{k-1}) = F_k \mathbf{x}_{k-1},  \ b_k( \mathbf{x}_k) =
      B_k \mathbf{x}_k \ \mathrm{and} \ \mathbf{\nu}_k \sim \mathcal{N}(0,Q_k)


    Notes
    -----
    In the Kalman filter, transition and control models must be linear.


    Raises
    ------
    ValueError
        If no :class:`~.TransitionModel` is specified.


    """

    transition_model = Property(LinearGaussianTransitionModel,
                                doc="The transition model to be used. "
                                    "Functions will throw a "
                                    ":class:`~.ValueError` if not specified. ")
    control_model = Property(LinearControlModel, default=None,
                             doc="The control model to be used. In the event "
                                 "that this is undefined the predictor will "
                                 "create a zero-effect linear "
                                 ":class:`~.ControlModel`.")

    def __init__(self, transition_model, control_model=None, *args, **kwargs):
        """Explicitly initialise the models and check for existence of
        transition model, and throw an error if not specified.
        """
        super().__init__(transition_model, control_model=None, *args, **kwargs)

        # Chuck an error if there's no transition model
        if self.transition_model is None:
            raise ValueError("A Predictor requires a state transition model")

        # If no control model insert a linear zero-effect one
        # TODO: Think about whether it's more efficient to leave this out
        ndims = self.transition_model.ndim_state
        if self.control_model is None:
            self.control_model = LinearControlModel(ndims, [],
                                                    np.zeros([ndims, 1]),
                                                    np.zeros([ndims, ndims]),
                                                    np.zeros([ndims, ndims]))

    def transition_matrix(self, **kwargs):
        """Return the transition matrix

        Parameters
        ----------
        **kwargs : various, optional
            These are passed to :class:`~.LinearGaussianTransitionModel`.
            :obj:`matrix()`

        Returns
        -------
        :class:`~.ndarray`
            The transition matrix, :math:`F_k`

        """
        return self.transition_model.matrix(**kwargs)

    def transition_function(self, prior, **kwargs):
        r"""Applies the linear transition function to a single vector in the
        absence of a control input, returns a single predicted state.

        Parameters
        ----------
        prior : :class:`~.State`
            The prior state, :math:`\mathbf{x}_{k-1}`

        **kwargs : various, optional
            These are passed to
            :class:`~.LinearGaussianTransitionModel`. :attr:`matrix()`

        Returns
        -------
        :class:`~.State`
            The predicted state

        """
        return self.transition_model.matrix(**kwargs) @ prior.state_vector

    @property
    def control_matrix(self):
        r"""Convenience function which returns the control matrix

        Returns
        -------
        :class:`~.ndarray`
            control matrix, :math:`B_k`

        """
        return self.control_model.matrix()

    def _predict_over_interval(self, prior, timestamp):
        """Private function to get the prediction interval (or None)

        Parameters
        ----------
        prior : :class:`~.State`
            The prior state

        timestamp : :class:`~.datetime`. :attr:`datetime`, optional
            The (current) timestamp

        Returns
        -------
        :class:`~.datetime`. :attr:`time_interval`
            time interval to predict over

        """

        # Deal with undefined timestamps
        if (timestamp is None) or (prior.timestamp is None):
            predict_over_interval = None
        else:
            predict_over_interval = timestamp - prior.timestamp

        return predict_over_interval

    @lru_cache()
    def predict(self, prior, timestamp=None, **kwargs):
        r"""The predict function

        Parameters
        ----------
        prior : :class:`~.State`
            :math:`\mathbf{x}_{k-1}`
        timestamp : :class:`~.datetime`. :attr:`datetime`, optional
            :math:`k`
        **kwargs :
            These are passed, via :class:`~.KalmanFilter`.
            :attr:`transition_function()` to
            :class:`~.LinearGaussianTransitionModel`. :attr:`matrix()`

        Returns
        -------
        :class:`~.State`
            :math:`\mathbf{x}_{k|k-1}`, the predicted state and the predicted
            state covariance :math:`P_{k|k-1}`

        """

        # Get the prediction interval
        predict_over_interval = self._predict_over_interval(prior, timestamp)

        # Prediction of the mean
        x_pred = self.transition_function(
            prior, time_interval=predict_over_interval, **kwargs) + \
            self.control_model.control_input()

        # As this is Kalman-like, the control model must be capable of
        # returning a control matrix (B)
        p_pred = self.transition_matrix(prior=prior,
                                        time_interval=predict_over_interval,
                                        **kwargs) @ \
            prior.covar @ \
            self.transition_matrix(prior=prior,
                                   time_interval=predict_over_interval,
                                   **kwargs).T + \
            self.transition_model.covar(
                     time_interval=predict_over_interval, **kwargs) + \
            self.control_matrix @ self.control_model.control_noise @ \
            self.control_matrix.T

        return GaussianStatePrediction(x_pred, p_pred, timestamp=timestamp)


class ExtendedKalmanPredictor(KalmanPredictor):
    """ExtendedKalmanPredictor class

    An implementation of the Extended Kalman Filter predictor. Here the
    transition and control functions may be non-linear, their transition and
    control matrices are approximated via Jacobian matrices. To this end the
    transition and control models, if non-linear, must be able to return the
    :attr:`jacobian()` function.

    """

    # In this version the models can be non-linear, but must have access to the
    # :attr:`jacobian()` function
    # TODO: Enforce the presence of :attr:`jacobian()`
    transition_model = Property(TransitionModel,
                                doc="The transition model to be used. "
                                    "Functions will throw a "
                                    ":class:`~.ValueError` if not specified.")
    control_model = Property(ControlModel, default=None,
                             doc="The control model to be used. If undefined "
                                 "the predictor will create a zero-effect "
                                 "linear :class:`~.ControlModel`.")

    def transition_matrix(self, prior, **kwargs):
        r"""Returns the transition matrix, a matrix if the model is linear, or
        approximated as Jacobian otherwise.

        Parameters
        ----------
        prior : :class:`~.State`
            :math:`\mathbf{x}_{k-1}`

        **kwargs : various, optional
            These are passed to :class:`~.TransitionModel`. :obj:`jacobian()`

        Returns
        -------
        transition matrix : :obj:`ndarray`
            The transition matrix, :math:`F_k`, if linear (i.e. the
            :attr:`matrix()` function exists), or
            :class:`~.TransitionModel`. :attr:`jacobian()` if not

        """
        if hasattr(self.transition_model, 'matrix'):
            return self.transition_model.matrix(**kwargs)
        else:
            return self.transition_model.jacobian(prior.state_vector, **kwargs)

    def transition_function(self, prior, **kwargs):
        r"""This is the application of :math:`f_k(\mathbf{x}_{k-1})`, the
        transition function, non-linear in general, in the absence of a control
        input

        Parameters
        ----------
        prior : :class:`~.State`
            The prior state, :math:`\mathbf{x}_{k-1}`

        **kwargs : various, optional
            These are passed to :class:`~.ExtendedKalmanFilter`.
            :attr:`function()`

        Returns
        -------
        :class:`~.State`
            The predicted state

        """
        return self.transition_model.function(prior.state_vector, noise=0,
                                              **kwargs)

    @property
    def control_matrix(self):
        r"""Returns the control input model matrix, :math:`B_k`, or its linear
        approximation via a Jacobian. The :class:`~.ControlModel`, if
        non-linear must therefore be capable of returning a :obj:`jacobian()`
        function.

        Returns
        -------
        :obj:`numpy.ndarray`
            The control model matrix, or its linear approximation
        """
        if hasattr(self.control_model, 'matrix'):
            return self.control_model.matrix()
        else:
            return self.control_model.jacobian(
                self.control_model.control_vector)


class UnscentedKalmanPredictor(KalmanPredictor):
    """UnscentedKalmanFilter class

    The predict is accomplished by calculating the sigma points from the
    Gaussian mean and covariance, then putting these through the (in general
    non-linear) transition function, then reconstructing the Gaussian.

    """
    # The models may be non-linear and non-differentiable
    transition_model = Property(TransitionModel,
                                doc="The transition model to be used. "
                                    "Functions will throw a "
                                    ":class:`~.ValueError` if not"
                                    " specified.")
    control_model = Property(ControlModel, default=None,
                             doc="The control model to be used. If undefined "
                                 "the predictor will create a zero-effect "
                                 "linear :class:~.`ControlModel`.")

    alpha = Property(float, default=0.5,
                     doc="Primary sigma point spread scaling parameter. "
                         "Typically 0.5.")
    beta = Property(float, default=2,
                    doc="Used to incorporate prior knowledge of the "
                        "distribution. If the true distribution is Gaussian, "
                        "the value of 2 is optimal.")
    kappa = Property(float, default=0, doc="Secondary spread scaling parameter"
                                           " (default is calculated as 3-Ns)")

    _time_interval = Property(datetime.timedelta, default=None,
                              doc="Hidden variable where time interval is "
                                  "optionally stored")

    def transition_and_control_function(self, prior_state_vector, **kwargs):
        r"""Returns the result of applying the transition and control functions
        for the unscented transform

        Parameters
        ----------
        prior_state_vector : :class:`~.State`
            Prior state vector
        **kwargs : various, optional
            These are passed to :class:`~.TransitionModel`.
            :attr:`function()`

        Returns
        -------
        :class:`numpy.ndarray`
            The combined, noiseless, effect of applying the transition and control

        """

        predict_over_interval = self._time_interval
        return self.transition_model.function(
            prior_state_vector, time_interval=predict_over_interval, noise=0,
            **kwargs) + self.control_model.control_input()

    @lru_cache()
    def predict(self, prior, timestamp=None, **kwargs):
        r"""The unscented version of the predict step

        Parameters
        ----------
        prior : :class:`~.State`
            Prior state, :math:`\mathbf{x}_{k-1}`
        timestamp : :class:`~.datetime`. :obj:`datetime`
            Time to transit to (:math:`k`)
        **kwargs : various, optional
            These are passed to :class:`~.TransitionModel`. :attr:`covar()`

        Returns
        -------
        :class:`~.GaussianStatePrediction`
            The predicted state :math:`\mathbf{x}_{k|k-1}` and the predicted
            state covariance :math:`P_{k|k-1}`

        """

        # Get the prediction interval
        predict_over_interval = self._predict_over_interval(prior, timestamp)
        # This ensures that function passed to unscented transform has the
        # correct time interval
        self._time_interval = predict_over_interval

        # The covariance on the transition model + the control model
        # TODO: Note that I'm not sure you can actually do this with the
        # TODO: covariances, i.e. sum them before calculating
        # TODO: the sigma points and then just sticking them into the
        # TODO: unscented transform, and I haven't checked the statistics.
        total_noise_covar = self.transition_model.covar(
            timestamp=timestamp, time_interval=predict_over_interval,
            **kwargs) + self.control_model.control_noise

        # Get the sigma points from the prior mean and covariance.
        sigma_points, mean_weights, covar_weights = gauss2sigma(
            prior.state_vector, prior.covar, self.alpha, self.beta, self.kappa)

        # Put these through the unscented transform, together with the total
        # covariance to get the parameters of the Gaussian
        x_pred, p_pred, _, _, _, _ = unscented_transform(
            sigma_points, mean_weights, covar_weights,
            self.transition_and_control_function, covar_noise=total_noise_covar
        )

        # and return a Gaussian state based on these parameters
        return GaussianStatePrediction(x_pred, p_pred, timestamp=timestamp)
