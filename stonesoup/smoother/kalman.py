# -*- coding: utf-8 -*-
import numpy as np

from .base import Smoother
from ..base import Property
from ..types.track import Track
from ..types.prediction import Prediction, GaussianStatePrediction
from ..types.update import Update, GaussianStateUpdate
from ..models.transition.base import TransitionModel
from ..models.transition.linear import LinearGaussianTransitionModel


class KalmanSmoother(Smoother):
    """
    The linear-Gaussian or Rauch-Tung-Striebel smoother, colloquially the Kalman smoother [1]_. The
    transition model is therefore linear-Gaussian. No control model is currently implemented.

    TODO: Include a control model

    The smooth function undertakes the backward algorithm on a :class:`~.Track()` object. This is
    done by starting at the final index in the track, :math:`K` and proceed from
    :math:`K \rightarrow 1` via:

    .. math::

        \mathbf{x}_{k+1|k} = F_{k+1} \mathbf{x}_k

        P_{k+1|k} = F_{k+1} P_k F_{k+1}^T + Q_{k+1}

        G_k = P_k F_{k+1}^T P_{k+1|k}^{-1}

        \mathbf{x}_{k}^s = \mathbf{x}_k + G_k (\mathbf{x}_{k+1}^s - \mathbf{x}_{k+1|k})

        P_k^s = P_k + G_k (P_{k+1}^s - P_{k+1|k}) G_k^T

    where :math:`\mathbf{x}_{K}^s = \mathbf{x}_{K}` and :math:`P_K^s = P_K`.


    References

    .. [1] S\"ark\"a S. 2013, Bayesian filtering and smoothing, Cambridge University Press

    """

    transition_model: LinearGaussianTransitionModel = Property(doc="The transition model to be used.")

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
                transition_matrix = self._transition_model(state).matrix(time_interval)

                ksmooth_gain = state.covar @ transition_matrix.T @ np.linalg.inv(prediction.covar)
                smooth_mean = state.state_vector + ksmooth_gain @ (prev_state.state_vector -
                                                                   prediction.state_vector)
                smooth_covar = state.covar + \
                    ksmooth_gain @ (prev_state.covar - prediction.covar) @ ksmooth_gain.T

                # Create a new type called SmoothedState?
                if isinstance(state, Update):
                    prev_state = Update.from_state(state, smooth_mean, smooth_covar,
                                                   timestamp=state.timestamp, hypothesis=state.hypothesis)
                elif isinstance(state, Prediction):
                    prev_state = Prediction.from_state(state, smooth_mean, smooth_covar,
                                                       timestamp=state.timestamp)

                smoothed_track.append(prev_state)

            # Save the predicted mean and covariance for the next (i.e. previous) timestep
            prediction = self._prediction(state)

        smoothed_track.reverse()
        return smoothed_track


class ExtendedKalmanSmoother(KalmanSmoother):
    """

    """
    transition_model: TransitionModel = Property(doc="The transition model to be used.")
