# -*- coding: utf-8 -*-
"""Test for smoother.lineargaussian"""
import numpy as np

from stonesoup.types import Track, GaussianState, CovarianceMatrix
from stonesoup.transitionmodel import LinearTransitionModel
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.smoother.lineargaussian import Backward


def test_backwards_smooother():
    """Linear Gaussian Backward Smoother test"""

    # Generate some data and apply Kalman filter.
    T = 6
    Q = CovarianceMatrix(0.2 * np.eye(2))
    R = CovarianceMatrix(0.01 * np.eye(2))
    Model = LinearTransitionModel(np.array([[1, 0.05], [0, 1]]))

    process_noise = [
        np.array([[-0.2646889 ],
                  [ 0.25171505]]),
        np.array([[-0.23481911],
                  [ 0.34099158]]),
        np.array([[-0.01526152],
                  [-0.48696623]]),
        np.array([[-0.21121374],
                  [0.42081589]]),
        np.array([[ 0.68543224],
                  [-0.25126595]])
    ]

    measurement_noise = [
        np.array([[0.20368294],
                  [-0.00502441]]),
        np.array([[-0.06915021],
                  [ 0.13709835]]),
        np.array([[-0.04640669],
                  [0.03396034]]),
        np.array([[0.11083243],
                  [-0.04758062]]),
        np.array([[0.05524205],
                  [-0.02261846]]),
        np.array([[0.2596889],
                  [-0.09494711]])
    ]

    truth = [np.array([[3.45], [1.02]])]
    detections = [GaussianState(truth[0]+measurement_noise[0], R)]

    for t in range(T-1):
        truth.append(Model.transition(detections[-1].state_vector) + process_noise[t])
        detections.append(GaussianState(truth[-1] + measurement_noise[t+1], R))

    # Filter
    track = Track()
    track.states.append(detections[0])
    new_state, innov = KalmanUpdater.update(track, detections[0])
    track.states[-1] = new_state

    kalman_predictor = KalmanPredictor(Model, Q)
    estimates = [detections[0]]

    for t in range(1, T):
        estimate = kalman_predictor.predict(track.state)
        track.states.append(estimate)
        estimates.append(estimate)
        new_state, innov = KalmanUpdater.update(track, detections[t])
        track.states[-1] = new_state

    # Smooth
    smoothed_track = Backward.batch_smooth(track, estimates, Model)

    assert np.allclose(smoothed_track.states[0].mean,
                       np.array([[3.64559832],[1.02400049]]))
    assert np.allclose(smoothed_track.states[0].covar,
                       np.array([[ 4.88348075e-03, -5.70201551e-06],
                                 [-5.70201551e-06, 4.88319446e-03]]))