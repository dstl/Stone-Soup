# -*- coding: utf-8 -*-
"""Tests the various Kalman-based smoothers. This test replicates that the exists/existed in
test_lineargaussian"""

import pytest
import numpy as np
from datetime import datetime, timedelta

from stonesoup.types.detection import Detection
from stonesoup.types.multihypothesis import MultipleHypothesis
from stonesoup.types.state import GaussianState
from stonesoup.types.prediction import GaussianStatePrediction
from stonesoup.types.track import Track
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.models.transition.linear import ConstantVelocity
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.types.update import GaussianStateUpdate
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.smoother.kalman import KalmanSmoother, ExtendedKalmanSmoother, \
    UnscentedKalmanSmoother


@pytest.fixture(
    params=[
        KalmanSmoother,  # Standard Kalman
        ExtendedKalmanSmoother,  # Extended Kalman
        UnscentedKalmanSmoother,  # Unscented Kalman
    ],
    ids=["standard", "extended", "unscented"]
)
def smoother_class(request):
    return request.param


@pytest.mark.parametrize("multi_hypothesis", [False, True],
                         ids=["SingleHypothesis", "MultipleHypothesis"])
def test_kalman_smoother(smoother_class, multi_hypothesis):

    # First create a track from some detections and then smooth - check the output.

    # Setup list of Detections
    start = datetime.now()
    times = [start + timedelta(seconds=i) for i in range(0, 5)]

    measurements = [
        np.array([[2.486559674128609]]),
        np.array([[2.424165626519697]]),
        np.array([[6.603176662762473]]),
        np.array([[9.329099124074590]]),
        np.array([[14.637975326666801]]),
    ]

    detections = [Detection(m, timestamp=timest) for m, timest in zip(measurements, times)]

    # Setup models.
    trans_model = ConstantVelocity(noise_diff_coeff=1)
    meas_model = LinearGaussian(ndim_state=2, mapping=[0], noise_covar=np.array([[0.4]]))

    # Tracking components
    predictor = KalmanPredictor(transition_model=trans_model)
    updater = KalmanUpdater(measurement_model=meas_model)

    # Prior
    cstate = GaussianState(np.ones([2, 1]), np.eye(2), timestamp=start)
    track = Track()

    for detection in detections:
        # Predict
        pred = predictor.predict(cstate, timestamp=detection.timestamp)
        # form hypothesis
        hypothesis = SingleHypothesis(pred, detection)
        # Update
        cstate = updater.update(hypothesis)
        # write to track
        if multi_hypothesis:
            cstate.hypothesis = MultipleHypothesis([hypothesis])
        track.append(cstate)

    smoother = smoother_class(transition_model=trans_model)
    smoothed_track = smoother.smooth(track)
    smoothed_state_vectors = [
        state.state_vector for state in smoothed_track]

    # Verify Values
    target_smoothed_vectors = [
        np.array([[1.688813974839928], [1.267196351952188]]),
        np.array([[3.307200214998506], [2.187167840595264]]),
        np.array([[6.130402001958210], [3.308896367021604]]),
        np.array([[9.821303658438408], [4.119557021638030]]),
        np.array([[14.257730973981149], [4.594862462495096]])
    ]

    assert np.allclose(smoothed_state_vectors, target_smoothed_vectors)

    # Check that a prediction is smoothable and that no error chucked
    # Also remove the transition model and use the one provided by the smoother
    track[1] = GaussianStatePrediction(pred.state_vector, pred.covar, timestamp=pred.timestamp)
    smoothed_track2 = smoother.smooth(track)
    assert isinstance(smoothed_track2[1], GaussianStatePrediction)

    # Check appropriate error chucked if not GaussianStatePrediction/Update
    track[-1] = detections[-1]
    with pytest.raises(TypeError):
        smoother._prediction(track[-1])


def test_multi_prediction_exception(smoother_class):
    """Check error raised with multi-hypothesis with multiple predictions"""
    start = datetime.now()
    track = Track()
    track.states = [
        GaussianStateUpdate(
            state_vector=[1],
            covar=[[1]],
            timestamp=start,
            hypothesis=MultipleHypothesis([
                SingleHypothesis(
                    GaussianStatePrediction([1], [[1]], timestamp=start), None),
                SingleHypothesis(
                    GaussianStatePrediction([2], [[1]], timestamp=start), None),
                ])
        ),
        GaussianStateUpdate(
            state_vector=[1],
            covar=[[1]],
            timestamp=start+timedelta(1),
            hypothesis=MultipleHypothesis([
                SingleHypothesis(
                    GaussianStatePrediction([1], [[1]], timestamp=start+timedelta(1)), None),
                SingleHypothesis(
                    GaussianStatePrediction([2], [[1]], timestamp=start+timedelta(1)), None),
            ])
        )
    ]
    smoother = smoother_class(transition_model=ConstantVelocity(1))
    with pytest.raises(
            ValueError, match="Track has MultipleHypothesis updates with multiple predictions"):
        smoother.smooth(track)
