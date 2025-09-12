import datetime

import numpy as np
import pytest

from stonesoup.feeder.bias import (
    TimeGaussianBiasFeeder,
    TranslationGaussianBiasFeeder, OrientationGaussianBiasFeeder,
    OrientationTranslationGaussianBiasFeeder)
from stonesoup.functions import build_rotation_matrix
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, RandomWalk
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.types.detection import Detection
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.state import GaussianState, StateVector


def make_dummy_detection(state_vector, measurement_model):
    det = Detection(
        state_vector,
        timestamp=datetime.datetime.now(),
        measurement_model=measurement_model
    )
    det.applied_bias = None
    return det


def make_dummy_measurement_model():
    class DummyMeasurementModel:
        mapping = [0, 1, 2]
        ndim_meas = 3
        translation_offset = StateVector([[0.], [0.], [0.]])
        rotation_offset = StateVector([[0.], [0.], [0.]])

        def function(self, state, noise=False, **kwargs):
            return build_rotation_matrix(self.rotation_offset) \
                @ (state.state_vector - self.translation_offset)

        def covar(self, *args, **kwargs):
            return np.eye(3)

        def jacobian(self, state, **kwargs):
            return np.eye(3, 6)

    return DummyMeasurementModel()


def make_dummy_transition_model():
    class DummyTransitionModel:
        def function(self, state, noise=False, time_interval=None, **kwargs):
            # Adds time_interval (seconds) to the state_vector
            if time_interval is None:
                time_interval = datetime.timedelta(seconds=0)
            return state.state_vector + time_interval.total_seconds()
    return DummyTransitionModel()


@pytest.fixture(params=[None, datetime.datetime(2025, 9, 9, 23, 59, 59)])
def bias_timestamp(request):
    return request.param


def test_translation_gaussian_bias_feeder_iter():
    # Setup feeder
    bias_prior = GaussianState(StateVector([[1.], [2.], [3.]]), np.eye(3))
    bias_predictor = KalmanPredictor(
        CombinedLinearGaussianTransitionModel([RandomWalk(1)] * 3))
    # Mock reader to yield a single time and detection
    measurement_model = make_dummy_measurement_model()
    detection = make_dummy_detection(StateVector([[10.], [20.], [30.]]), measurement_model)
    feeder = TranslationGaussianBiasFeeder(
        reader=[(datetime.datetime(2025, 9, 10), [detection])],
        bias_prior=bias_prior,
        bias_predictor=bias_predictor,
    )
    # Iterate over feeder
    for time, detections in feeder:
        # Bias should be applied to detection
        for det in detections:
            assert np.allclose(det.applied_bias, feeder.bias)
            # The measurement model's translation_offset should be updated
            assert np.allclose(
                det.measurement_model.translation_offset,
                -feeder.bias
            )


def test_translation_gaussian_bias_feeder_update_bias(bias_timestamp):
    # Setup feeder
    bias_prior = GaussianState(StateVector([[0.], [0.], [0.]]), np.eye(3) * 10, bias_timestamp)
    bias_predictor = KalmanPredictor(
        CombinedLinearGaussianTransitionModel([RandomWalk(1)] * 3))
    feeder = TranslationGaussianBiasFeeder(
        reader=None,
        bias_prior=bias_prior,
        bias_predictor=bias_predictor,
    )
    # Create dummy prediction and measurement
    measurement_model = make_dummy_measurement_model()
    pred = GaussianState(
        StateVector([[10.], [20.], [30.]]),
        np.eye(3),
        timestamp=datetime.datetime(2025, 9, 10)
    )
    meas = Detection(
        StateVector([[11.], [21.], [31.]]),
        timestamp=datetime.datetime(2025, 9, 10),
        measurement_model=measurement_model
    )
    meas.applied_bias = np.zeros((3, 1))
    # Create hypothesis
    hyp = SingleHypothesis(pred, meas)
    # Call update_bias
    updates = feeder.update_bias([hyp])
    # The bias_state should be updated (should not be the same as initial)...
    assert not np.allclose(feeder.bias_state.state_vector, bias_prior.state_vector)
    # and should be around 0.8 ± 0.1
    assert np.allclose(feeder.bias_state.state_vector, [[0.8], [0.8], [0.8]], atol=0.1)
    # The returned updates should match the updated state shape
    assert updates[0].state_vector.shape == pred.state_vector.shape


def test_orientation_gaussian_bias_feeder_iter():
    bias_prior = GaussianState(StateVector([[0.], [0.], [np.pi/16]]), np.eye(3))
    bias_predictor = KalmanPredictor(
        CombinedLinearGaussianTransitionModel([RandomWalk(1)] * 3))
    measurement_model = make_dummy_measurement_model()
    detection = make_dummy_detection(StateVector([[10.], [20.], [30.]]), measurement_model)
    feeder = OrientationGaussianBiasFeeder(
        reader=[(datetime.datetime(2025, 9, 10), [detection])],
        bias_prior=bias_prior,
        bias_predictor=bias_predictor,
    )
    for time, detections in feeder:
        for det in detections:
            assert np.allclose(det.applied_bias, feeder.bias)
            assert np.allclose(
                det.measurement_model.rotation_offset,
                StateVector([[0.], [0.], [0.]]) - feeder.bias
            )
            rotated = build_rotation_matrix(det.measurement_model.rotation_offset) \
                @ (det.state_vector - det.measurement_model.translation_offset)
            expected_rotated = build_rotation_matrix(-feeder.bias) \
                @ (det.state_vector - det.measurement_model.translation_offset)
            assert np.allclose(rotated, expected_rotated)


def test_orientation_gaussian_bias_feeder_update_bias(bias_timestamp):
    bias_prior = GaussianState(
        StateVector([[0.], [0.], [0.]]), np.diag([1e-6, 1e-6, 1]), bias_timestamp)
    bias_predictor = KalmanPredictor(
        CombinedLinearGaussianTransitionModel([RandomWalk(1e-6)] * 3))
    feeder = OrientationGaussianBiasFeeder(
        reader=None,
        bias_prior=bias_prior,
        bias_predictor=bias_predictor,
        max_bias=StateVector([[0.], [0.], [np.pi]])
    )
    measurement_model = make_dummy_measurement_model()
    pred = GaussianState(
        StateVector([[10.], [20.], [30.]]),
        np.eye(3),
        timestamp=datetime.datetime(2025, 9, 10)
    )
    meas = Detection(
        build_rotation_matrix(np.array([[0.], [0.], [-np.pi/16]]))
        @ StateVector([[10.], [20.], [30.]]),
        timestamp=datetime.datetime(2025, 9, 10),
        measurement_model=measurement_model
    )
    meas.applied_bias = np.zeros((3, 1))
    hyp = SingleHypothesis(pred, meas)
    updates = feeder.update_bias([hyp])
    assert not np.allclose(feeder.bias_state.state_vector, bias_prior.state_vector)
    assert np.allclose(feeder.bias_state.state_vector, [[0.], [0.], [np.pi/16]], atol=0.05)
    assert updates[0].state_vector.shape == pred.state_vector.shape


def test_orientation_translation_gaussian_bias_feeder_iter():
    bias_prior = GaussianState(
        StateVector([[0.], [0.], [np.pi/16], [1.], [2.], [3.]]), np.diag([1e-6, 1e-6, 1, 3, 3, 3]))
    bias_predictor = KalmanPredictor(
        CombinedLinearGaussianTransitionModel([RandomWalk(1e-6)]*3 + [RandomWalk(1)]*3))
    measurement_model = make_dummy_measurement_model()
    detection = make_dummy_detection(StateVector([[10.], [20.], [30.]]), measurement_model)
    feeder = OrientationTranslationGaussianBiasFeeder(
        reader=[(datetime.datetime(2025, 9, 10), [detection])],
        bias_prior=bias_prior,
        bias_predictor=bias_predictor,
    )
    for time, detections in feeder:
        for det in detections:
            assert np.allclose(det.applied_bias, feeder.bias)
            assert np.allclose(
                np.vstack([
                    det.measurement_model.rotation_offset,
                    det.measurement_model.translation_offset]),
                -feeder.bias
            )
            rotated = build_rotation_matrix(det.measurement_model.rotation_offset) \
                @ (det.state_vector - det.measurement_model.translation_offset)
            expected_rotated = build_rotation_matrix(-feeder.bias) \
                @ (det.state_vector - det.measurement_model.translation_offset)
            assert np.allclose(rotated, expected_rotated)


def test_orientation_translation_gaussian_bias_feeder_update_bias(bias_timestamp):
    bias_prior = GaussianState(
        StateVector([[0.], [0.], [np.pi/16], [1.], [2.], [3.]]),
        np.diag([1e-6, 1e-6, 1, 3, 3, 3]),
        bias_timestamp)
    bias_predictor = KalmanPredictor(
        CombinedLinearGaussianTransitionModel([RandomWalk(1e-6)]*3 + [RandomWalk(1)]*3))
    feeder = OrientationTranslationGaussianBiasFeeder(
        reader=None,
        bias_prior=bias_prior,
        bias_predictor=bias_predictor,
    )
    measurement_model = make_dummy_measurement_model()
    hyps = []
    for n in range(-10, 11):  # A lot ok unknowns so a few detections needed
        pred = GaussianState(
            StateVector([[10.*n], [20.*n], [30.*n]]),
            np.eye(3),
            timestamp=datetime.datetime(2025, 9, 10)
        )
        meas = Detection(
            build_rotation_matrix(np.array([[0.], [0.], [-np.pi/16]]))
            @ (pred.state_vector+StateVector([[1.], [1.], [1.]])),
            timestamp=datetime.datetime(2025, 9, 10),
            measurement_model=measurement_model
        )
        meas.applied_bias = np.zeros((6, 1))
        hyp = SingleHypothesis(pred, meas)
        hyps.append(hyp)
    updates = feeder.update_bias(hyps)
    assert not np.allclose(feeder.bias_state.state_vector, bias_prior.state_vector)
    assert np.allclose(
        feeder.bias_state.state_vector,
        [[0.], [0.], [np.pi/16], [1.], [1.], [1.]],
        atol=0.1)
    assert updates[0].state_vector.shape == pred.state_vector.shape


def test_time_gaussian_bias_feeder_iter():
    bias_prior = GaussianState(StateVector([[0.5]]), np.eye(1))
    bias_predictor = KalmanPredictor(
        CombinedLinearGaussianTransitionModel([RandomWalk(1)]))
    measurement_model = make_dummy_measurement_model()
    detection = make_dummy_detection(StateVector([[42.]]), measurement_model)
    orig_timestamp = detection.timestamp
    feeder = TimeGaussianBiasFeeder(
        reader=[(datetime.datetime(2025, 9, 10), [detection])],
        transition_model=None,
        bias_prior=bias_prior,
        bias_predictor=bias_predictor,
    )
    for time, detections in feeder:
        for det in detections:
            assert np.allclose(det.applied_bias, feeder.bias)
            expected_timestamp = orig_timestamp - datetime.timedelta(seconds=feeder.bias[0])
            assert detection.timestamp == expected_timestamp


def test_time_gaussian_bias_feeder_update_bias(bias_timestamp):
    bias_prior = GaussianState(StateVector([[0.5]]), np.eye(1) * 10, bias_timestamp)
    bias_predictor = KalmanPredictor(
        CombinedLinearGaussianTransitionModel([RandomWalk(1e-3)]))
    feeder = TimeGaussianBiasFeeder(
        reader=None,
        transition_model=make_dummy_transition_model(),
        bias_prior=bias_prior,
        bias_predictor=bias_predictor,
    )
    measurement_model = make_dummy_measurement_model()
    pred = GaussianState(
        StateVector([[41.5]]),
        np.eye(1),
        timestamp=datetime.datetime(2025, 9, 10)
    )
    meas = Detection(
        StateVector([[41.]]),
        timestamp=datetime.datetime(2025, 9, 10),
        measurement_model=measurement_model
    )
    meas.applied_bias = StateVector([[0.5]])
    hyp = SingleHypothesis(pred, meas)
    updates = feeder.update_bias([hyp])
    assert not np.allclose(feeder.bias_state.state_vector, bias_prior.state_vector)
    assert np.allclose(feeder.bias_state.state_vector, [[1.0]], atol=0.1)
    assert updates[0].state_vector.shape == pred.state_vector.shape
