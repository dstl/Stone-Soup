import copy
import datetime
from functools import partial

import numpy as np
import pytest

from ..bias import GaussianBiasUpdater
from ...models.measurement.bias import (
    TimeBiasModelWrapper, TranslationBiasModelWrapper,
    OrientationBiasModelWrapper, OrientationTranslationBiasModelWrapper)
from ...functions import build_rotation_matrix
from ...models.transition.linear import CombinedLinearGaussianTransitionModel, RandomWalk
from ...predictor.kalman import KalmanPredictor
from ...types.detection import Detection
from ...types.hypothesis import SingleHypothesis
from ...types.state import GaussianState, StateVector
from ...types.track import Track


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


@pytest.fixture(params=[None, datetime.datetime(2025, 9, 10)])
def bias_timestamp(request):
    return request.param


def test_translation_gaussian_bias_feeder_update_bias(bias_timestamp):
    # Setup feeder
    bias_prior = GaussianState(StateVector([[0.], [0.], [0.]]), np.eye(3) * 10, bias_timestamp)
    bias_predictor = KalmanPredictor(
        CombinedLinearGaussianTransitionModel([RandomWalk(1)] * 3))
    updater = GaussianBiasUpdater(
        bias_track=Track([copy.copy(bias_prior)]),
        bias_predictor=bias_predictor,
        bias_model_wrapper=TranslationBiasModelWrapper,
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
    meas.measurement_model.applied_bias = np.zeros((3, 1))

    pred_meas = updater.predict_measurement(pred, measurement_model)
    unbias_pred_meas = updater.updater.predict_measurement(pred, measurement_model)
    assert np.sum(np.trace(pred_meas.covar)) > np.sum(np.trace(unbias_pred_meas.covar))

    # Create hypothesis
    hyp = SingleHypothesis(pred, meas)
    # Call update_bias
    updates = updater.update([hyp])
    # The bias_state should be updated (should not be the same as initial)...
    assert not np.allclose(updater.bias_track.state_vector, bias_prior.state_vector)
    # and should be around 0.8 ± 0.1
    assert np.allclose(updater.bias_track.state_vector, [[0.8], [0.8], [0.8]], atol=0.1)
    # The returned updates should match the updated state shape
    assert updates[0].state_vector.shape == pred.state_vector.shape


def test_orientation_gaussian_bias_feeder_update_bias(bias_timestamp):
    bias_prior = GaussianState(
        StateVector([[0.], [0.], [0.]]), np.diag([1e-6, 1e-6, 1]), bias_timestamp)
    bias_predictor = KalmanPredictor(
        CombinedLinearGaussianTransitionModel([RandomWalk(1e-6)] * 3))
    updater = GaussianBiasUpdater(
        bias_track=Track([copy.copy(bias_prior)]),
        bias_predictor=bias_predictor,
        bias_model_wrapper=OrientationBiasModelWrapper,
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
    meas.measurement_model.applied_bias = np.zeros((3, 1))

    pred_meas = updater.predict_measurement(pred, measurement_model)
    unbias_pred_meas = updater.updater.predict_measurement(pred, measurement_model)
    assert np.sum(np.trace(pred_meas.covar)) > np.sum(np.trace(unbias_pred_meas.covar))

    hyp = SingleHypothesis(pred, meas)
    updates = updater.update([hyp])
    assert not np.allclose(updater.bias_track.state_vector, bias_prior.state_vector)
    assert np.allclose(updater.bias_track.state_vector, [[0.], [0.], [np.pi/16]], atol=0.05)
    assert updates[0].state_vector.shape == pred.state_vector.shape


def test_orientation_translation_gaussian_bias_feeder_update_bias(bias_timestamp):
    bias_prior = GaussianState(
        StateVector([[0.], [0.], [np.pi/16], [1.], [2.], [3.]]),
        np.diag([1e-6, 1e-6, 1, 3, 3, 3]),
        bias_timestamp)
    bias_predictor = KalmanPredictor(
        CombinedLinearGaussianTransitionModel([RandomWalk(1e-6)]*3 + [RandomWalk(1)]*3))
    updater = GaussianBiasUpdater(
        bias_track=Track([copy.copy(bias_prior)]),
        bias_predictor=bias_predictor,
        bias_model_wrapper=OrientationTranslationBiasModelWrapper
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
        meas.measurement_model.applied_bias = np.zeros((6, 1))
        hyp = SingleHypothesis(pred, meas)
        hyps.append(hyp)
    updates = updater.update(hyps)
    assert not np.allclose(updater.bias_track.state_vector, bias_prior.state_vector)
    assert np.allclose(
        updater.bias_track.state_vector,
        [[0.], [0.], [np.pi/16], [1.], [1.], [1.]],
        atol=0.1)
    assert updates[0].state_vector.shape == pred.state_vector.shape


def test_time_gaussian_bias_feeder_update_bias(bias_timestamp):
    bias_prior = GaussianState(StateVector([[0.5]]), np.eye(1) * 10, bias_timestamp)
    bias_predictor = KalmanPredictor(
        CombinedLinearGaussianTransitionModel([RandomWalk(1e-3)]))
    updater = GaussianBiasUpdater(
        bias_track=Track([copy.copy(bias_prior)]),
        bias_predictor=bias_predictor,
        bias_model_wrapper=partial(
            TimeBiasModelWrapper, transition_model=make_dummy_transition_model())
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
    meas.measurement_model.applied_bias = StateVector([[0.5]])

    pred_meas = updater.predict_measurement(pred, measurement_model)
    unbias_pred_meas = updater.updater.predict_measurement(pred, measurement_model)
    assert np.sum(np.trace(pred_meas.covar)) > np.sum(np.trace(unbias_pred_meas.covar))

    hyp = SingleHypothesis(pred, meas)
    updates = updater.update([hyp])
    assert not np.allclose(updater.bias_track.state_vector, bias_prior.state_vector)
    assert np.allclose(updater.bias_track.state_vector, [[1.0]], atol=0.1)
    assert updates[0].state_vector.shape == pred.state_vector.shape
