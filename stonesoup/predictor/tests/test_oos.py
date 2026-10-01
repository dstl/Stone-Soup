import datetime

import numpy as np
import pytest

from ..kalman import KalmanPredictor
from ..oos import OOSPredictorWrapper, get_hypothesis, get_past_states, get_prior_state
from ...models.measurement.linear import LinearGaussian
from ...models.transition.linear import ConstantVelocity
from ...smoother.kalman import KalmanSmoother
from ...types.detection import Detection, MissedDetection
from ...types.hypothesis import SingleHypothesis
from ...types.multihypothesis import MultipleHypothesis
from ...types.prediction import GaussianStatePrediction
from ...types.state import GaussianState
from ...types.track import Track
from ...types.update import GaussianStateUpdate, Update
from ...updater.kalman import KalmanUpdater

start_time = datetime.datetime(2025, 1, 1)


def time(seconds):
    return start_time + datetime.timedelta(seconds=seconds)


@pytest.fixture()
def transition_model():
    return ConstantVelocity(0.1)


@pytest.fixture()
def predictor(transition_model):
    return KalmanPredictor(transition_model)


@pytest.fixture()
def updater():
    return KalmanUpdater(
        LinearGaussian(ndim_state=2, mapping=[0], noise_covar=np.array([[1.]])))


@pytest.fixture()
def track(predictor, updater):
    """Track with initial state at 0 seconds, followed by updates at 1 to 5 seconds, such that
    index in the track is the same as the time in seconds"""
    track = Track([GaussianState([[0.], [1.]], np.diag([1., 1.]), time(0))])
    for k in range(1, 6):
        prediction = predictor.predict(track.state, time(k))
        detection = Detection([[k + 0.1*(-1)**k]], timestamp=time(k))
        track.append(updater.update(SingleHypothesis(prediction, detection)))
    return track


def assert_states_equal(state1, state2):
    assert state1.timestamp == state2.timestamp
    assert np.allclose(state1.state_vector, state2.state_vector)
    assert np.allclose(state1.covar, state2.covar)


def test_get_hypothesis():
    prediction = GaussianStatePrediction([[0.], [1.]], np.diag([1., 1.]), time(0))
    null_hypothesis = SingleHypothesis(prediction, MissedDetection(timestamp=time(0)))
    hypothesis1 = SingleHypothesis(prediction, Detection([[0.]], timestamp=time(0)))
    hypothesis2 = SingleHypothesis(prediction, Detection([[1.]], timestamp=time(0)))

    # Single hypothesis is returned as is, regardless of filter
    assert get_hypothesis(hypothesis1) is hypothesis1
    assert get_hypothesis(hypothesis1, lambda hypothesis: not hypothesis) is hypothesis1
    assert get_hypothesis(null_hypothesis, bool) is null_hypothesis

    # First single hypothesis matching filter is returned from multiple hypothesis
    multiple_hypothesis = MultipleHypothesis([null_hypothesis, hypothesis1, hypothesis2])
    assert get_hypothesis(multiple_hypothesis) is null_hypothesis
    assert get_hypothesis(multiple_hypothesis, bool) is hypothesis1
    assert get_hypothesis(multiple_hypothesis, lambda hypothesis: not hypothesis) \
        is null_hypothesis

    # None if no hypothesis matches the filter
    assert get_hypothesis(MultipleHypothesis([hypothesis1, hypothesis2]),
                          lambda hypothesis: not hypothesis) is None


def test_get_prior_state(track, predictor):
    # Update: prior of the prediction in the hypothesis
    assert get_prior_state(track[5]) is track[4]
    assert get_prior_state(track[1]) is track[0]

    # Prediction: prior of the prediction
    prediction = predictor.predict(track[5], time(6))
    assert get_prior_state(prediction) is track[5]

    # Update with multiple hypothesis: prior of the prediction in the first hypothesis
    multiple_hypothesis = MultipleHypothesis([
        SingleHypothesis(prediction, MissedDetection(timestamp=time(6))),
        SingleHypothesis(prediction, Detection([[6.]], timestamp=time(6)))])
    update = Update.from_state(prediction, hypothesis=multiple_hypothesis)
    assert isinstance(update, GaussianStateUpdate)
    assert get_prior_state(update) is track[5]

    # Update without prediction e.g. from an initiator
    update = Update.from_state(
        prediction, hypothesis=SingleHypothesis(None, Detection([[6.]], timestamp=time(6))))
    assert get_prior_state(update) is None

    # Prediction without prior
    assert get_prior_state(GaussianStatePrediction([[0.], [1.]], np.diag([1., 1.]))) is None

    # Other states have no prior
    assert get_prior_state(track[0]) is None
    assert get_prior_state(None) is None


def test_get_past_states(track, predictor):
    past_states = list(get_past_states(track[5]))
    assert len(past_states) == 6
    assert all(state1 is state2 for state1, state2 in zip(past_states, reversed(track)))

    # From a prediction
    prediction = predictor.predict(track[5], time(6))
    past_states = list(get_past_states(prediction))
    assert len(past_states) == 7
    assert past_states[0] is prediction
    assert all(state1 is state2 for state1, state2 in zip(past_states[1:], reversed(track)))

    # State with no prior
    assert list(get_past_states(track[0])) == [track[0]]
    assert list(get_past_states(None)) == []


@pytest.mark.parametrize('prior_type', ['state', 'track'])
def test_predict_in_sequence(track, predictor, prior_type):
    oos_predictor = OOSPredictorWrapper(predictor)
    prior = track if prior_type == 'track' else track[5]

    prediction = oos_predictor.predict(prior, time(6))
    assert_states_equal(prediction, predictor.predict(track[5], time(6)))
    assert prediction.prior is track[5]
    assert prediction.prior.orig_prior is track[5]

    # Same time as latest state
    prediction = oos_predictor.predict(prior, time(5))
    assert_states_equal(prediction, track[5])
    assert prediction.prior is track[5]
    assert prediction.prior.orig_prior is track[5]


@pytest.mark.parametrize('prior_type', ['state', 'track'])
@pytest.mark.parametrize(
    'seconds, prior_index',
    [
        (4.5, 4),
        (2.5, 2),
        (3, 3),  # Same time as a past state
        (0.5, 0),  # Before first update
        (0, 0),
    ])
def test_predict_out_of_sequence(track, predictor, prior_type, seconds, prior_index):
    oos_predictor = OOSPredictorWrapper(predictor)
    prior = track if prior_type == 'track' else track[5]

    prediction = oos_predictor.predict(prior, time(seconds))

    # Prediction is from the most recent state at or before the timestamp
    assert_states_equal(prediction, predictor.predict(track[prior_index], time(seconds)))
    assert prediction.prior is track[prior_index]
    # with reference to the latest state
    assert prediction.prior.orig_prior is track[5]


def test_predict_before_earliest_state(track, predictor):
    oos_predictor = OOSPredictorWrapper(predictor)
    with pytest.raises(ValueError, match="Cannot predict"):
        oos_predictor.predict(track[5], time(-1))

    # Different model, so can check the backward predictor is the one used
    backward_predictor = KalmanPredictor(ConstantVelocity(0))
    oos_predictor = OOSPredictorWrapper(predictor, backward_predictor=backward_predictor)
    prediction = oos_predictor.predict(track[5], time(-1))
    assert_states_equal(prediction, backward_predictor.predict(track[0], time(-1)))
    assert prediction.prior is track[0]
    assert prediction.prior.orig_prior is track[5]

    # Backward predictor not used when a state is available to predict forward from
    prediction = oos_predictor.predict(track[5], time(2.5))
    assert_states_equal(prediction, predictor.predict(track[2], time(2.5)))
    prediction = oos_predictor.predict(track[5], time(6))
    assert_states_equal(prediction, predictor.predict(track[5], time(6)))


def test_predict_smoother(track, predictor, transition_model):
    smoother = KalmanSmoother(transition_model)
    oos_predictor = OOSPredictorWrapper(predictor, smoother=smoother)

    # Out of sequence: prediction is from smoothed state, using the later states
    prediction = oos_predictor.predict(track[5], time(2.5))
    smoothed_state = smoother.smooth(track[2:])[0]
    assert smoothed_state.timestamp == time(2)
    assert not np.allclose(smoothed_state.state_vector, track[2].state_vector)
    assert_states_equal(prediction, predictor.predict(smoothed_state, time(2.5)))
    assert not np.allclose(
        prediction.state_vector, predictor.predict(track[2], time(2.5)).state_vector)
    # but the prior is still the original state
    assert prediction.prior is track[2]
    assert prediction.prior.orig_prior is track[5]

    # In sequence: no later states to smooth with
    prediction = oos_predictor.predict(track[5], time(6))
    assert_states_equal(prediction, predictor.predict(track[5], time(6)))
    assert prediction.prior is track[5]

    # Before first update: no update to smooth, so predict from the initial state
    prediction = oos_predictor.predict(track[5], time(0.5))
    assert_states_equal(prediction, predictor.predict(track[0], time(0.5)))
    assert prediction.prior is track[0]
    assert prediction.prior.orig_prior is track[5]


def test_predict_smoother_missed_detection(track, predictor, updater, transition_model):
    # Add missed detections at 6 and 7 seconds, followed by an update
    track.append(predictor.predict(track[5], time(6)))
    track.append(predictor.predict(track[6], time(7)))
    prediction = predictor.predict(track[7], time(8))
    track.append(updater.update(
        SingleHypothesis(prediction, Detection([[8.5]], timestamp=time(8)))))

    smoother = KalmanSmoother(transition_model)
    oos_predictor = OOSPredictorWrapper(predictor, smoother=smoother)

    # Smoothed from most recent update before the predictions
    prediction = oos_predictor.predict(track[8], time(6.5))
    smoothed_state = smoother.smooth(Track([track[5], track[8]]))[0]
    assert smoothed_state.timestamp == time(5)
    assert_states_equal(prediction, predictor.predict(smoothed_state, time(6.5)))
    assert prediction.prior is track[6]
    assert prediction.prior.orig_prior is track[8]
