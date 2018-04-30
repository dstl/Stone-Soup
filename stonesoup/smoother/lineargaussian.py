# -*- coding: utf-8 -*-
import copy

import numpy as np

from .base import Smoother
from ..types import GaussianState


class Backward(Smoother):
    """ Backwards component of a fixed-interval forward-backward smoother for a Linear Gaussian State Space Model.
    """

    @staticmethod
    def batch_smooth(filtered_track, estimates, linear_transition_model):
        """ Apply smoothing to a track of filtered estimates e.g. output from a Kalman Filter.

        Parameters
        ----------
        filtered_track:
            Track object consisting a of GaussianState objects.
        estimates:
            List of T GaussianState objects corresponding to the Track.
        linear_transition_model:
            LinearTransitionModel

        Returns
        -------
        smoothed_track:
            Track object containing smoothed GaussianStates.
        """
        track_length = len(filtered_track.states)
        if track_length != len(estimates):
            raise ValueError(
                "filtered_track.states and estimates should have the same length")

        penultimate_time_index = track_length - 2

        smoothed_track = copy.deepcopy(filtered_track)

        # Iterating backwards from the penultimate state, to the first state.
        for t in range(penultimate_time_index,-1,-1):
            smoothed_track.states[t] = Backward.smooth(filtered_track.states[t],
                                                       estimates[t+1],
                                                       smoothed_track.states[t+1],
                                                       linear_transition_model)

        return smoothed_track

    @staticmethod
    def smooth(filtered_state_t, predicted_state_tplus1, smoothed_state_tplus1, linear_transition_model):
        """ Deduce smoothed distribution p(x_t | y_1:T) from p(x_t | y_1:t), p(x_t+1 | y_1:t) and p(x_t+1 | y_1:T).

        Parameters
        ----------
        filtered_state_t:
            GaussianState object representing the filtered state at time t.

        predicted_state_tplus1:
            GaussianState object representing the prediction (at timestep t) of the state at time t+1.

        smoothed_state_tplus1:
            GaussianState object representing the smoothed state at time t+1.

        linear_transition_model:
            LinearTransitionModel

        Returns
        -------
        GaussianState object
        """
        A = linear_transition_model.transition_matrix

        x = filtered_state_t.mean
        V = filtered_state_t.covar
        x_predict = predicted_state_tplus1.mean
        V_predict = predicted_state_tplus1.covar
        x_tplus1 = smoothed_state_tplus1.mean
        V_tplus1 = smoothed_state_tplus1.covar

        smoother_gain = V @ A.T @ np.linalg.inv(V_predict)

        x_smoothed = x + smoother_gain @ (x_tplus1 - x_predict)
        V_smoothed = V + smoother_gain @ (V_tplus1 - V_predict) @ smoother_gain.T

        return GaussianState(x_smoothed, V_smoothed)