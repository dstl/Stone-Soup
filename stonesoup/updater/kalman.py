# -*- coding: utf-8 -*-

import numpy as np

from .base import Updater


class KalmanUpdater(Updater):
    """Simple Kalman Filter

    Perform measurement update step in the standard Kalman Filter.
    """

    @staticmethod
    def update(target_state, state_covar, target_meas,
               meas_covar, meas_mat=None):
        if meas_mat is None:
            meas_mat = np.eye(len(target_meas), len(target_state))

        innov = target_meas - meas_mat @ target_state

        innov_covar = meas_covar + meas_mat @ state_covar @ meas_mat.T
        gain = state_covar @ meas_mat.T @ np.linalg.inv(innov_covar)

        updated_state = target_state + gain @ innov

        temp = gain @ meas_mat
        temp = np.eye(*temp.shape) - temp
        updated_state_covar = (
            temp @ state_covar @ temp.T + gain @ meas_covar @ gain.T)

        return updated_state, updated_state_covar, innov, innov_covar
