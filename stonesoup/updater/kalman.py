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


class SqrtKalmanUpdater(Updater):
    """Square Root Kalman Filter

    Perform measurement update step in the square root Kalman Filter.
    """

    @staticmethod
    def update(target_state, state_covar, target_meas,
               meas_covar, meas_mat=None):
        # state_covar and meas_covar are lower triangular matrices
        if meas_mat is None:
            meas_mat = np.eye(len(target_meas), len(target_state))

        innov = target_meas - meas_mat @ target_state

        Pxz = state_covar @ state_covar.T @ meas_mat.T
        innov_covar = SqrtKalmanUpdater.tria(
            np.concatenate(((meas_mat @ state_covar), meas_covar), axis=1))
        gain = (Pxz @ np.linalg.inv(innov_covar.T)) @ np.linalg.inv(innov_covar)

        updated_state = target_state + gain @ innov

        temp = gain @ meas_mat
        updated_state_covar = SqrtKalmanUpdater.tria(np.concatenate(
            (((np.eye(*temp.shape) - temp) @ state_covar), (gain @ meas_covar)),
            axis=1))

        return updated_state, updated_state_covar, innov, innov_covar

    @staticmethod
    # Possible abstract method?
    def tria(matrix):
        """Square Root Matrix Triangularization

        Given a rectangular square root matrix obtain a square lower-triangular
        square root matrix
        """
        if not isinstance(matrix, np.matrixlib.defmatrix.matrix):
            matrix = np.matrix(matrix)

        [_, upper_triangular] = np.linalg.qr(matrix.T)
        lower_triangular = upper_triangular.T

        # Bug? 'Make diagonal elements positive'
        lower_triangular = np.abs(lower_triangular)
        # np.fill_diagonal(lower_triangular, np.abs(lower_triangular.diagonal()))

        return np.array(lower_triangular)

