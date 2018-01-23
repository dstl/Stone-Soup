# -*- coding: utf-8 -*-

import numpy as np

from .base import Updater
from ..types import StateVector
from ..functions import tria


class KalmanUpdater(Updater):
    """Simple Kalman Filter

    Perform measurement update step in the standard Kalman Filter.
    """

    @staticmethod
    def update(track, detection, meas_mat=None):
        if meas_mat is None:
            meas_mat = np.eye(len(detection.state), len(track.state))

        innov = detection.state - meas_mat @ track.state

        innov_covar = detection.covar + meas_mat @ track.covar @ meas_mat.T
        gain = track.covar @ meas_mat.T @ np.linalg.inv(innov_covar)

        updated_state = track.state + gain @ innov

        temp = gain @ meas_mat
        temp = np.eye(*temp.shape) - temp
        updated_state_covar = (
            temp @ track.covar @ temp.T + gain @ detection.covar @ gain.T)

        return (
            StateVector(updated_state, updated_state_covar),
            StateVector(innov, innov_covar))



class SqrtKalmanUpdater(Updater):
    """Square Root Kalman Filter

    Perform measurement update step in the square root Kalman Filter.
    """

    @staticmethod
    def update(track, detection, meas_mat=None):
        # track.covar and detection.covar are lower triangular matrices
        if meas_mat is None:
            meas_mat = np.eye(len(detection.state), len(track.state))

        innov = detection.state - meas_mat @ track.state

        Pxz = track.covar @ track.covar.T @ meas_mat.T
        innov_covar = tria(
            np.concatenate(((meas_mat @ track.covar), detection.covar), axis=1))
        gain = (Pxz @ np.linalg.inv(innov_covar.T)) @ np.linalg.inv(innov_covar)

        updated_state = track.state + gain @ innov

        temp = gain @ meas_mat
        updated_state_covar = tria(np.concatenate(
            (((np.eye(*temp.shape) - temp) @ track.covar),
             (gain @ detection.covar)),
            axis=1))

        return (
            StateVector(updated_state, updated_state_covar),
            StateVector(innov, innov_covar))

