# -*- coding: utf-8 -*-
import numpy as np

from .base import Smoother
from ..base import Property
from ..types.track import Track
from ..types.prediction import Prediction, GaussianStatePrediction
from ..types.update import Update, GaussianStateUpdate
from ..models.base import LinearModel
from ..models.transition.base import TransitionModel
from ..models.transition.linear import LinearGaussianTransitionModel


class KalmanSmoother(Smoother):
    r"""
    The linear-Gaussian or Rauch-Tung-Striebel smoother, colloquially the Kalman smoother [1]_. The
    transition model is therefore linear-Gaussian. No control model is currently implemented.

    TODO: Include a control model

    The smooth function undertakes the backward algorithm on a :class:`~.Track()` object. This is
    done by starting at the final index in the track, :math:`K` and proceeding from
    :math:`K \rightarrow 1` via:

    .. math::

        \mathbf{x}_{k|k-1} &= F_{k} \mathbf{x}_{k-1}

        P_{k|k-1} &= F_{k} P_{k-1} F_{k}^T + Q_{k}

        G_k &= P_{k-1} F_{k}^T P_{k|k-1}^{-1}

        \mathbf{x}_{k-1}^s &= \mathbf{x}_{k-1} + G_k (\mathbf{x}_{k}^s - \mathbf{x}_{k|k-1})

        P_{k-1}^s &= P_{k-1} + G_k (P_{k}^s - P_{k|k-1}) G_k^T

    where :math:`\mathbf{x}_{K}^s = \mathbf{x}_{K}` and :math:`P_K^s = P_K`.

    The predicted state vector and covariance are retrieved from the Track via predicted state or
    updated state via the links therein. Note that this means that the first two equations are not
    calculated, the results merely retrieved. This smoother is therefore strictly Kalman in the
    backward portion. The prediction might have come by any number of means. If present, the
    transition model (providing :math:`F` and :math:`Q`) in the prediction is used. This allows for
    a dynamic transition model (i.e. one that changes with :math:`k`). Otherwise, the (static)
    transition model is used, defined on smoother initialisation.

    References

    .. [1] Särkä S. 2013, Bayesian filtering and smoothing, Cambridge University Press

    """

    transition_model: LinearGaussianTransitionModel = Property(
        doc="The transition model. The :meth:`smooth` function will initially look for a "
            "transition model in the prediction. If that is not found then this one is used.")

    def _prediction(self, state):
        """ Return the predicted state, either from the prediction directly, or from the attached
        hypothesis if the queried state is an Update. If not a :class:`~.GaussianStatePrediction`
        or :class:`~.GaussianStateUpdate` a :class:`~.TypeError` is thrown.

        Parameters
        ----------
        state : :class:`~.GaussianStatePrediction` or :class:`~.GaussianStateUpdate`

        Returns
        -------
         : :class:`~.GaussianStatePrediction`
            The prediction associated with the prediction (i.e. itself), or the prediction from the
            hypothesis used to generate an update.
        """
        if isinstance(state, GaussianStatePrediction):
            return state
        elif isinstance(state, GaussianStateUpdate):
            return state.hypothesis.prediction
        else:
            raise TypeError("States must be GaussianStatePredictions or GaussianStateUpdates.")

    def _transition_model(self, state):
        """ If it exists, return the transition model from the prediction associated with input
        state. If that doesn't exist then use the (static) transition model defined by the
        smoother.

        Parameters
        ----------
        state : :class:`~.GaussianStatePrediction` or :class:`~.GaussianStateUpdate`

        Returns
        -------
         : :class:`~.TransitionModel`
            The transition model to be associated with state
        """
        # Is there a transition model linked to the prediction?
        if hasattr(self._prediction(state), "transition_model"):
            transition_model = self._prediction(state).transition_model
        else:  # No? Return the class attribute
            transition_model = self.transition_model

        return transition_model

    def _transition_matrix(self, state, **kwargs):
        """ Return the transition matrix

        Parameters
        ----------
        state : :class:`~.State`
            The input state (to check for a linked prediction)
        **kwargs
            These are passed to the :meth:`matrix()` function

        Returns
        -------
         : :class:`numpy.ndarray`
            The transition matrix
        """
        return self._transition_model(state).matrix(**kwargs)

    def smooth(self, track):
        """
        Perform the backward recursion to smooth the track.

        Parameters
        ----------
        track : :class:`~.Track`
            The input track.

        Returns
        -------
         : :class:`~.Track`
            Smoothed track

        """
        firststate = True
        smoothed_track = Track()
        for state in reversed(track):

            if firststate:
                prev_state = state
                smoothed_track.append(state)
                firststate = False
            else:
                # Delta t
                time_interval = prev_state.timestamp - state.timestamp

                # Get the transition model matrix
                transition_matrix = self._transition_matrix(state, time_interval=time_interval)

                ksmooth_gain = state.covar @ transition_matrix.T @ np.linalg.inv(prediction.covar)
                smooth_mean = state.state_vector + ksmooth_gain @ (prev_state.state_vector -
                                                                   prediction.state_vector)
                smooth_covar = state.covar + \
                    ksmooth_gain @ (prev_state.covar - prediction.covar) @ ksmooth_gain.T

                # Create a new type called SmoothedState?
                if isinstance(state, Update):
                    prev_state = Update.from_state(state, smooth_mean, smooth_covar,
                                                   timestamp=state.timestamp,
                                                   hypothesis=state.hypothesis)
                elif isinstance(state, Prediction):
                    prev_state = Prediction.from_state(state, smooth_mean, smooth_covar,
                                                       timestamp=state.timestamp)

                smoothed_track.append(prev_state)

            # Save the predicted mean and covariance for the next (i.e. previous) timestep
            prediction = self._prediction(state)

        smoothed_track.reverse()
        return smoothed_track


class ExtendedKalmanSmoother(KalmanSmoother):
    r""" The extended version of the Kalman smoother. The equations are modified slightly,
    analogously to the extended Kalman filter,

    .. math::

        \mathbf{x}_{k|k-1} &= f_{k} (\mathbf{x}_{k-1})

        F_k &\approx J_f (\mathbf{x}_{k-1})

    where :math:`J_f (\mathbf{x}_{k-1})` is the Jacobian matrix evaluated at
    :math:`\mathbf{x}_{k-1}`. The rest of the calculation proceeds as with the Kalman smoother.

    In fact the first equation isn't calculated -- it's presumed to have been undertaken by a
    filter when building the track; similarly for the predicted covariance. In practice, the only
    difference between this and the Kalman smoother is in the use of the linearised transition
    matrix to calculate the smoothing gain.

    """
    transition_model: TransitionModel = Property(doc="The transition model to be used.")

    def _transition_matrix(self, state, **kwargs):
        r"""Returns the transition matrix, a matrix if the model is linear, or
        approximated as Jacobian otherwise.

        Parameters
        ----------
        state : :class:`~.State`
            :math:`\mathbf{x}_{k-1}`
        **kwargs : various, optional
            These are passed to :meth:`~.TransitionModel.matrix` or
            :meth:`~.TransitionModel.jacobian`

        Returns
        -------
        : :class:`numpy.ndarray`
            The transition matrix, :math:`F_k`, if linear (i.e.
            :meth:`TransitionModel.matrix` exists, or
            :meth:`~.TransitionModel.jacobian` if not)
        """
        if isinstance(self._transition_model(state), LinearModel):
            return self._transition_model(state).matrix(**kwargs)
        else:
            return self._transition_model(state).jacobian(state, **kwargs)
