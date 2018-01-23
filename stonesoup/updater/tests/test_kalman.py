# -*- coding: utf-8 -*-
"""Test for updater.kalman module"""
import numpy as np

from ...types import Track, Detection, StateVector
from ..kalman import KalmanUpdater, SqrtKalmanUpdater


def test_sqrtkalman():
    """Square Root Kalman Updater test"""

    # TODO: Better test data
    estimate = StateVector(np.array([[-1.8513], [0.9994], [0], [0]]) * 1e4,
                           np.array([[2.2128, 0, 0, 0],
                                     [0.0002, 2.2130, 0, 0],
                                     [0.3897, -0.00004, 0.0128, 0],
                                     [0, 0.3897, 0.0013, 0.0135]]) * 1e3)
    detection = Detection(np.array([[2.4378], [1.0072]]) * 1e4,
                          np.array([[19.8607, 0], [-35.8829, 23.1799]]))
    track = Track()
    track.estimates.append(estimate)
    state_vector, innov = SqrtKalmanUpdater.update(track, detection)

    assert np.allclose(state_vector.state, np.array([
        [2.4375], [1.0078], [0.7553], [0.0014]]) * 1e4, 1)
    assert np.allclose(state_vector.covar, np.array([
        [19.8573, 0, 0, 0],
        [-35.8728, 23.1786, 0, 0],
        [3.4975, -0.0004, 12.8034, 0],
        [-6.3167, 4.0812, 1.2637, 13.5487]]), 0.1)

    assert np.allclose(innov.state, np.array([[4.2891], [0.0078]]) * 1e4, 1)
    assert np.allclose(innov.covar, np.array([
        [2.2129, 0],
        [-0.0001, 2.2134]]) * 1e3, 1)
