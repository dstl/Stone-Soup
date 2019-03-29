# -*- coding: utf-8 -*-
import copy

import numpy as np

from .base import Smoother
from ..types.state import GaussianState


class Backward(Smoother):
    """ Backwards component of a fixed-interval forward-backward smoother for
    a Linear Gaussian State Space Model.
    """

    def track_smooth(self, filtered_track, estimates):
        """ Apply smoothing to a track of filtered estimates.

        Parameters
        ----------
        filtered_track : :class:`Track`
            Track object consisting a of GaussianState objects.
        estimates : :obj:`list` of :class:`GaussianState`
            List of T GaussianState objects corresponding to the Track.

        Returns
        -------
        smoothed_track : :class:`Track`
            Track object containing smoothed GaussianStates.
        """
        track_length = len(filtered_track)
        if track_length != len(estimates):
            raise ValueError(
                "filtered_track and estimates should have the same "
                "length")

        penultimate_index = track_length - 2

        smoothed_track = copy.deepcopy(filtered_track)

        # Iterating backwards from the penultimate state, to the first state.
        for t in range(penultimate_index, -1, -1):
            smoothed_track[t] = self.smooth(filtered_track[t],
                                            estimates[t+1],
                                            smoothed_track[t+1])

        return smoothed_track

    def smooth(self, filtered_state_t, predicted_state_tplus1,
               smoothed_state_tplus1):
        """ Deduce smoothed distribution :math:`p(x_{t} | y_{1:T})` from
        :math:`p(x_{t} | y_{1:t})`, :math:`p(x_{t+1} | y_{1:t})`
        and :math:`p(x_{t+1} | y_{1:T})`.

        Parameters
        ----------
        filtered_state_t : :class:`GaussianState`
            Filtered state at time t.
        predicted_state_tplus1 : :class:`GaussianState`
            Prediction (at timestep t), of the state at time t+1.
        smoothed_state_tplus1 : :class:`GaussianState`
            Smoothed state at time t+1.

        Returns
        -------
        : :class:`GaussianState`
        """
        t = filtered_state_t.timestamp
        t_delta = smoothed_state_tplus1.timestamp - t
        A = self.transition_model.matrix(time_interval=t_delta)

        x = filtered_state_t.mean
        V = filtered_state_t.covar
        x_predict = predicted_state_tplus1.mean
        V_predict = predicted_state_tplus1.covar
        x_tplus1 = smoothed_state_tplus1.mean
        V_tplus1 = smoothed_state_tplus1.covar

        smoother_gain = V @ A.T @ np.linalg.inv(V_predict)

        x_smoothed = x + smoother_gain@(x_tplus1 - x_predict)
        V_smoothed = V + smoother_gain@(V_tplus1 - V_predict)@smoother_gain.T

        return GaussianState(x_smoothed, V_smoothed, timestamp=t)
