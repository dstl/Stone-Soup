# -*- coding: utf-8 -*-
"""Test for smoother.lineargaussian"""
import datetime

import numpy as np

from stonesoup.types.detection import Detection
from stonesoup.types.state import GaussianState
from stonesoup.types.track import Track
from stonesoup.models.transition.linear import ConstantVelocity
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.smoother.lineargaussian import Backward


def test_backwards_smoother():
    """Linear Gaussian Backward Smoother test"""

    # Setup list of Detections
    T = 5
    measurements = [
        np.array([[2.486559674128609]]),
        np.array([[2.424165626519697]]),
        np.array([[6.603176662762473]]),
        np.array([[9.329099124074590]]),
        np.array([[14.637975326666801]]),
    ]
    detections = [Detection(m) for m in measurements]

    t_0 = datetime.datetime.now()
    t_delta = datetime.timedelta(0, 1)
    t = [t_0]
    for n in range(T-1):
        t.append(t[-1] + t_delta)
    for n in range(T):
        detections[n].timestamp = t[n]

    # Setup models.
    initial_state = GaussianState(np.ones([2, 1]), np.eye(2), timestamp=t_0)

    trans_model = ConstantVelocity(noise_diff_coeff=1)
    meas_model = LinearGaussian(ndim_state=2, mapping=[0],
                                noise_covar=np.array([[0.4]]))

    estimates = [initial_state]

    # Filter Initial Detection
    track = Track()
    track.states.append(initial_state)
    predictor = KalmanPredictor(transition_model=trans_model)
    updater = KalmanUpdater(measurement_model=meas_model)
    new_state = updater.update(track.state, detections[0])
    track.states[0] = new_state

    # Filter Remaining Detections.
    for t in range(1, T):
        time = detections[t].timestamp
        state_pred = predictor.predict(track.state, timestamp=time)
        estimates.append(state_pred)
        state_post = updater.update(state_pred, detections[t])
        track.states.append(state_post)

    # Smooth Track
    smoother = Backward(transition_model=trans_model)
    smoothed_track = smoother.track_smooth(track, estimates)

    smoothed_state_vectors = [
        state.state_vector for state in smoothed_track.states]

    # Verify Values
    target_smoothed_vectors = [
        np.array([[1.688813974839928], [1.267196351952188]]),
        np.array([[3.307200214998506], [2.187167840595264]]),
        np.array([[6.130402001958210], [3.308896367021604]]),
        np.array([[9.821303658438408], [4.119557021638030]]),
        np.array([[14.257730973981149], [4.594862462495096]])
    ]

    assert np.allclose(smoothed_state_vectors, target_smoothed_vectors)
