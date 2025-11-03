import datetime

import numpy as np

from stonesoup.feeder.bias import (
    TimeBiasFeeder,
    TranslationBiasFeeder, OrientationBiasFeeder,
    OrientationTranslationBiasFeeder)
from stonesoup.functions import build_rotation_matrix
from stonesoup.types.detection import Detection
from stonesoup.types.state import GaussianState, StateVector
from stonesoup.types.track import Track


def make_dummy_detection(state_vector, measurement_model):
    det = Detection(
        state_vector,
        timestamp=datetime.datetime.now(),
        measurement_model=measurement_model
    )
    det.measurement_model.applied_bias = None
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


def test_translation_gaussian_bias_feeder_iter():
    # Setup feeder
    bias_state = GaussianState(StateVector([[1.], [2.], [3.]]), np.eye(3))
    # Mock reader to yield a single time and detection
    measurement_model = make_dummy_measurement_model()
    detection = make_dummy_detection(StateVector([[10.], [20.], [30.]]), measurement_model)
    feeder = TranslationBiasFeeder(
        reader=[(datetime.datetime(2025, 9, 10), [detection])],
        bias_track=Track([bias_state]),
    )
    # Iterate over feeder
    for time, detections in feeder:
        # Bias should be applied to detection
        for det in detections:
            assert np.allclose(det.measurement_model.applied_bias, feeder.bias)
            # The measurement model's translation_offset should be updated
            assert np.allclose(
                det.measurement_model.translation_offset,
                -feeder.bias
            )


def test_orientation_gaussian_bias_feeder_iter():
    bias_state = GaussianState(StateVector([[0.], [0.], [np.pi/16]]), np.eye(3))
    measurement_model = make_dummy_measurement_model()
    detection = make_dummy_detection(StateVector([[10.], [20.], [30.]]), measurement_model)
    feeder = OrientationBiasFeeder(
        reader=[(datetime.datetime(2025, 9, 10), [detection])],
        bias_track=Track([bias_state]),
    )
    for time, detections in feeder:
        for det in detections:
            assert np.allclose(det.measurement_model.applied_bias, feeder.bias)
            assert np.allclose(
                det.measurement_model.rotation_offset,
                StateVector([[0.], [0.], [0.]]) - feeder.bias
            )
            rotated = build_rotation_matrix(det.measurement_model.rotation_offset) \
                @ (det.state_vector - det.measurement_model.translation_offset)
            expected_rotated = build_rotation_matrix(-feeder.bias) \
                @ (det.state_vector - det.measurement_model.translation_offset)
            assert np.allclose(rotated, expected_rotated)


def test_orientation_translation_gaussian_bias_feeder_iter():
    bias_state = GaussianState(
        StateVector([[0.], [0.], [np.pi/16], [1.], [2.], [3.]]), np.diag([1e-6, 1e-6, 1, 3, 3, 3]))
    measurement_model = make_dummy_measurement_model()
    detection = make_dummy_detection(StateVector([[10.], [20.], [30.]]), measurement_model)
    feeder = OrientationTranslationBiasFeeder(
        reader=[(datetime.datetime(2025, 9, 10), [detection])],
        bias_track=Track([bias_state]),
    )
    for time, detections in feeder:
        for det in detections:
            assert np.allclose(det.measurement_model.applied_bias, feeder.bias)
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


def test_time_gaussian_bias_feeder_iter():
    bias_state = GaussianState(StateVector([[0.5]]), np.eye(1))
    measurement_model = make_dummy_measurement_model()
    detection = make_dummy_detection(StateVector([[42.]]), measurement_model)
    orig_timestamp = detection.timestamp
    feeder = TimeBiasFeeder(
        reader=[(datetime.datetime(2025, 9, 10), [detection])],
        bias_track=Track([bias_state]),
    )
    for time, detections in feeder:
        for det in detections:
            assert np.allclose(det.measurement_model.applied_bias, feeder.bias)
            expected_timestamp = orig_timestamp - datetime.timedelta(seconds=feeder.bias[0])
            assert detection.timestamp == expected_timestamp
