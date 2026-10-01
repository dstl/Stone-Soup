import datetime
import gc

import numpy as np
import pytest

from ..kalman import ExtendedKalmanUpdater, KalmanUpdater
from ..oos import OOSUpdaterWrapper
from ..probability import PDAUpdater
from ...hypothesiser.probability import PDAHypothesiser
from ...initiator.simple import SimpleMeasurementInitiator, SinglePointInitiator
from ...models.measurement.linear import LinearGaussian
from ...models.transition.linear import ConstantVelocity
from ...predictor.kalman import KalmanPredictor
from ...predictor.oos import OOSPredictorWrapper, get_past_states
from ...types.detection import Detection, MissedDetection
from ...types.hypothesis import SingleHypothesis
from ...types.multihypothesis import MultipleHypothesis
from ...types.prediction import Prediction
from ...types.state import GaussianState
from ...types.update import Update

start_time = datetime.datetime(2025, 1, 1)


def time(seconds):
    return start_time + datetime.timedelta(seconds=seconds)


def seconds(state):
    return (state.timestamp - start_time).total_seconds()


def detection(seconds):
    return Detection([[seconds + 0.5*np.sin(3*seconds)]], timestamp=time(seconds))


@pytest.fixture(scope='module', autouse=True)
def gc_freeze():
    """Freeze existing objects, so that the garbage collection run in these tests is quicker"""
    gc.collect()
    gc.freeze()
    yield
    gc.unfreeze()


def clear_caches():
    """Clear caches which hold references to states, and run garbage collection (as states may
    be in reference cycles), to ensure tests aren't reliant on states being kept alive by these"""
    OOSPredictorWrapper._predict_states.cache_clear()
    OOSPredictorWrapper.predict.__wrapped__.cache_clear()
    KalmanPredictor.predict.__wrapped__.cache_clear()
    KalmanUpdater.predict_measurement.cache_clear()
    ExtendedKalmanUpdater.predict_measurement.cache_clear()
    gc.collect()


def run(predictor, updater, state, order, missed=()):
    """Run OOS predictor and updater for detections at each of the times in seconds in `order`,
    with missed detections at times in `missed`. Only a reference to the latest state is kept,
    so history is reliant on references held by the updater."""
    for seconds_ in order:
        prediction = predictor.predict(state, time(seconds_))
        if seconds_ not in missed:
            measurement = detection(seconds_)
        else:
            measurement = MissedDetection(timestamp=time(seconds_))
        state = updater.update(SingleHypothesis(prediction, measurement))
        del prediction
        clear_caches()
    return state


def run_expected(predictor, updater, state, order, missed=()):
    """Run standard predictor and updater for detections at each of the times in seconds in
    `order` (which is sorted), with missed detections at times in `missed`. Returns list of all
    states, most recent first."""
    states = [state]
    for seconds_ in sorted(order):
        prediction = predictor.predict(states[0], time(seconds_))
        if seconds_ not in missed:
            states.insert(0, updater.update(SingleHypothesis(prediction, detection(seconds_))))
        else:
            states.insert(0, prediction)
    return states


def assert_states_equal(state1, state2):
    assert state1.timestamp == state2.timestamp
    assert np.allclose(state1.state_vector, state2.state_vector)
    assert np.allclose(state1.covar, state2.covar)


def assert_history_equal(state, expected_states):
    past_states = list(get_past_states(state))
    assert len(past_states) == len(expected_states)
    for past_state, expected_state in zip(past_states, expected_states):
        assert isinstance(past_state, Update) == isinstance(expected_state, Update)
        assert isinstance(past_state, Prediction) == isinstance(expected_state, Prediction)
        assert_states_equal(past_state, expected_state)


@pytest.fixture()
def transition_model():
    return ConstantVelocity(0.1)


@pytest.fixture()
def measurement_model():
    return LinearGaussian(ndim_state=2, mapping=[0], noise_covar=np.array([[1.]]))


@pytest.fixture()
def predictor(transition_model):
    return KalmanPredictor(transition_model)


@pytest.fixture()
def updater(measurement_model):
    return KalmanUpdater(measurement_model)


@pytest.fixture()
def oos_predictor(predictor):
    # Backward predictor with different model, without process noise
    return OOSPredictorWrapper(predictor, backward_predictor=KalmanPredictor(ConstantVelocity(0)))


@pytest.fixture()
def oos_updater(updater, oos_predictor):
    return OOSUpdaterWrapper(updater, oos_predictor)


@pytest.fixture()
def prior():
    return GaussianState([[0.], [1.]], np.diag([1., 1.]), time(0))


def test_predict_measurement(oos_updater, updater, predictor, prior, measurement_model):
    prediction = predictor.predict(prior, time(1))

    measurement_prediction = oos_updater.predict_measurement(prediction)
    expected = updater.predict_measurement(prediction)
    assert np.allclose(measurement_prediction.state_vector, expected.state_vector)
    assert np.allclose(measurement_prediction.covar, expected.covar)

    measurement_prediction = oos_updater.predict_measurement(
        prediction, measurement_model, measurement_noise=False)
    expected = updater.predict_measurement(prediction, measurement_model, measurement_noise=False)
    assert np.allclose(measurement_prediction.state_vector, expected.state_vector)
    assert np.allclose(measurement_prediction.covar, expected.covar)


def test_get_detections(predictor, prior):
    prediction = predictor.predict(prior, time(1))
    null_hypothesis = SingleHypothesis(prediction, MissedDetection(timestamp=time(1)))
    hypothesis1 = SingleHypothesis(prediction, detection(1))
    hypothesis2 = SingleHypothesis(prediction, detection(1))

    assert OOSUpdaterWrapper._get_detections(null_hypothesis) == set()
    assert OOSUpdaterWrapper._get_detections(hypothesis1) == {hypothesis1.measurement}
    assert OOSUpdaterWrapper._get_detections(
        MultipleHypothesis([null_hypothesis, hypothesis1, hypothesis2])) \
        == {hypothesis1.measurement, hypothesis2.measurement}
    assert OOSUpdaterWrapper._get_detections(MultipleHypothesis([null_hypothesis])) == set()


def test_in_sequence(oos_predictor, oos_updater, predictor, updater, prior):
    expected_states = run_expected(predictor, updater, prior, range(1, 6))

    oos_state = prior
    for seconds_ in range(1, 6):
        oos_state = run(oos_predictor, oos_updater, oos_state, [seconds_])

        assert isinstance(oos_state, Update)
        assert oos_state.hypothesis.measurement.timestamp == time(seconds_)
        assert_states_equal(oos_state, expected_states[-1 - seconds_])

    assert [seconds(past_state) for past_state in get_past_states(oos_state)] \
        == [5, 4, 3, 2, 1, 0]
    assert_history_equal(oos_state, expected_states)


@pytest.mark.parametrize(
    'order',
    [
        [1, 2, 4, 5, 3],  # Single out of sequence
        [1, 2, 5, 6, 3, 4],  # Two out of sequence, second after reprocessed state
        [1, 2, 5, 6, 4, 3],  # Two out of sequence, second before reprocessed state
        [1, 4, 2, 5, 3, 6],  # Alternating
        [3, 4, 5, 1, 2, 6],  # Before first update
        [6, 5, 4, 3, 2, 1],  # Reverse order
        [1, 2, 4, 5, 2.5, 4.5, 3],  # Non-uniform time intervals
    ],
    ids=str)
def test_out_of_sequence(oos_predictor, oos_updater, predictor, updater, prior, order):
    oos_state = run(oos_predictor, oos_updater, prior, order)
    expected_states = run_expected(predictor, updater, prior, order)

    # Estimate is for latest time, and same as if detections processed in sequence
    assert oos_state.timestamp == time(max(order))
    assert_states_equal(oos_state, expected_states[0])

    # History is reprocessed, and same as if detections processed in sequence
    assert [seconds(past_state) for past_state in get_past_states(oos_state)] \
        == sorted(order, reverse=True) + [0]
    assert_history_equal(oos_state, expected_states)
    assert list(get_past_states(oos_state))[-1] is prior


@pytest.mark.parametrize(
    'order, missed',
    [
        ([1, 2, 3, 4, 5], {5}),  # In sequence missed detection
        ([1, 2, 3, 4, 5], {2, 3}),
        ([1, 2, 4, 5, 3], {3}),  # Out of sequence missed detection
        ([1, 2, 4, 5, 3], {4}),  # Reprocessing a missed detection
        ([1, 2, 4, 5, 3], {3, 4}),
        ([1, 2, 5, 6, 3, 4], {3, 5}),
    ],
    ids=str)
def test_missed_detection(oos_predictor, oos_updater, predictor, updater, prior, order, missed):
    oos_state = run(oos_predictor, oos_updater, prior, order, missed)
    expected_states = run_expected(predictor, updater, prior, order, missed)

    assert oos_state.timestamp == time(max(order))
    assert_states_equal(oos_state, expected_states[0])

    # Missed detections are predictions in the history
    past_states = list(get_past_states(oos_state))
    assert [seconds(past_state) for past_state in past_states] \
        == sorted(order, reverse=True) + [0]
    for past_state in past_states[:-1]:
        if seconds(past_state) in missed:
            assert isinstance(past_state, Prediction)
        else:
            assert isinstance(past_state, Update)
    assert_history_equal(oos_state, expected_states)


def test_before_earliest_state(oos_predictor, oos_updater, prior):
    state = run(oos_predictor, oos_updater, prior, [1, 2, 3])
    prediction = oos_predictor.predict(state, time(-1))

    # Missed detection provides no new information, so latest state returned
    post = oos_updater.update(
        SingleHypothesis(prediction, MissedDetection(timestamp=time(-1))))
    assert post is state

    # Detection, but without an initiator
    with pytest.raises(RuntimeError, match="no initiator"):
        oos_updater.update(SingleHypothesis(prediction, detection(-1)))


@pytest.mark.parametrize('initiator_class', [SimpleMeasurementInitiator, SinglePointInitiator])
def test_before_earliest_state_initiator(
        oos_predictor, updater, predictor, prior, measurement_model, initiator_class):
    initiator = initiator_class(
        GaussianState([[0.], [1.]], np.diag([10., 1.])), measurement_model)
    oos_updater = OOSUpdaterWrapper(updater, oos_predictor, initiator=initiator)

    # Track which starts with state from an initiator, at 2 seconds
    initial_state = initiator.initiate({detection(2)}, time(2)).pop().state
    state = run(oos_predictor, oos_updater, initial_state, [3, 4])
    assert [seconds(past_state) for past_state in get_past_states(state)] == [4, 3, 2]

    # Detection from before start of the track, so initiate from this, and reprocess the rest
    oos_state = run(oos_predictor, oos_updater, state, [1])
    assert oos_state.timestamp == time(4)
    assert [seconds(past_state) for past_state in get_past_states(oos_state)] == [4, 3, 2, 1]

    expected_initial_state = initiator.initiate({detection(1)}, time(1)).pop().state
    expected_states = run_expected(predictor, updater, expected_initial_state, [2, 3, 4])
    assert_history_equal(oos_state, expected_states)

    # And again, with missed detection in the history
    oos_state = run(oos_predictor, oos_updater, oos_state, [5, 6, 0], missed={5})
    assert oos_state.timestamp == time(6)
    assert [seconds(past_state) for past_state in get_past_states(oos_state)] \
        == [6, 5, 4, 3, 2, 1, 0]

    expected_initial_state = initiator.initiate({detection(0)}, time(0)).pop().state
    expected_states = run_expected(
        predictor, updater, expected_initial_state, [1, 2, 3, 4, 5, 6], missed={5})
    assert_history_equal(oos_state, expected_states)


def test_before_earliest_state_initiator_unexpected_state(
        oos_predictor, updater, prior, measurement_model):
    initiator = SimpleMeasurementInitiator(
        GaussianState([[0.], [1.]], np.diag([10., 1.])), measurement_model)
    oos_updater = OOSUpdaterWrapper(updater, oos_predictor, initiator=initiator)

    # Track starts with a state which isn't an update or a prediction, so can't be reprocessed
    state = run(oos_predictor, oos_updater, prior, [1, 2, 3])
    prediction = oos_predictor.predict(state, time(-1))
    with pytest.raises(TypeError, match="Unexpected state type"):
        oos_updater.update(SingleHypothesis(prediction, detection(-1)))


def run_pda(hypothesiser, updater, state, order):
    states = [state]
    for seconds_ in order:
        hypotheses = hypothesiser.hypothesise(
            states[0], {detection(seconds_)}, time(seconds_))
        states.insert(0, updater.update(hypotheses))
    return states


@pytest.mark.parametrize(
    'order',
    [
        [1, 2, 4, 5, 3],
        [1, 2, 5, 6, 3, 4],
        [5, 4, 3, 2, 1],
    ],
    ids=str)
def test_hypothesiser(oos_predictor, predictor, prior, measurement_model, order):
    # Use GM method, which calculates measurement predictions, required when reprocessing
    updater = PDAUpdater(measurement_model, gm_method=True)
    hypothesiser = PDAHypothesiser(
        predictor, updater, clutter_spatial_density=0.1, prob_detect=0.9)
    oos_hypothesiser = PDAHypothesiser(
        oos_predictor, updater, clutter_spatial_density=0.1, prob_detect=0.9)

    expected_states = run_pda(hypothesiser, updater, prior, sorted(order))

    # With hypothesiser, association probabilities are recalculated when reprocessing, so same
    # as if detections processed in sequence
    oos_updater = OOSUpdaterWrapper(updater, oos_predictor, hypothesiser=oos_hypothesiser)
    oos_state = run_pda(oos_hypothesiser, oos_updater, prior, order)[0]
    assert [seconds(past_state) for past_state in get_past_states(oos_state)] \
        == sorted(order, reverse=True) + [0]
    assert_history_equal(oos_state, expected_states)
    for past_state, expected_past_state in zip(
            list(get_past_states(oos_state))[:-1], expected_states):
        assert isinstance(past_state.hypothesis, MultipleHypothesis)
        assert np.allclose(
            sorted(float(hypothesis.probability) for hypothesis in past_state.hypothesis),
            sorted(float(hypothesis.probability) for hypothesis in expected_past_state.hypothesis))

    # Without hypothesiser, existing association probabilities are used when reprocessing, so
    # result differs
    oos_updater = OOSUpdaterWrapper(updater, oos_predictor)
    oos_state = run_pda(oos_hypothesiser, oos_updater, prior, order)[0]
    assert oos_state.timestamp == expected_states[0].timestamp
    assert [seconds(past_state) for past_state in get_past_states(oos_state)] \
        == sorted(order, reverse=True) + [0]
    assert not np.allclose(oos_state.state_vector, expected_states[0].state_vector)


def test_min_history(oos_predictor, updater, predictor, prior):
    # Without min history, all history is kept
    oos_updater = OOSUpdaterWrapper(updater, oos_predictor)
    oos_state = run(oos_predictor, oos_updater, prior, range(1, 11))
    assert [seconds(past_state) for past_state in get_past_states(oos_state)] \
        == [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

    # With min history, history kept back to the most recent state at or before min history
    oos_updater = OOSUpdaterWrapper(
        updater, oos_predictor, min_history=datetime.timedelta(seconds=3))
    oos_state = run(oos_predictor, oos_updater, prior, range(1, 11))
    assert [seconds(past_state) for past_state in get_past_states(oos_state)] == [10, 9, 8, 7]

    # History shorter than, or same as, min history is all kept
    for num_steps in (2, 3):
        oos_state = run(oos_predictor, oos_updater, prior, range(1, num_steps + 1))
        assert [seconds(past_state) for past_state in get_past_states(oos_state)] \
            == list(range(num_steps, -1, -1))

    oos_updater = OOSUpdaterWrapper(
        updater, oos_predictor, min_history=datetime.timedelta(seconds=2.5))
    oos_state = run(oos_predictor, oos_updater, prior, range(1, 11))
    assert [seconds(past_state) for past_state in get_past_states(oos_state)] == [10, 9, 8, 7]

    # including where there are missed detections
    oos_state = run(oos_predictor, oos_updater, prior, range(1, 11), missed={7, 8})
    assert [seconds(past_state) for past_state in get_past_states(oos_state)] == [10, 9, 8, 7]


def test_min_history_multiple_hypothesis(oos_predictor, prior, measurement_model):
    updater = PDAUpdater(measurement_model, gm_method=True)
    hypothesiser = PDAHypothesiser(
        oos_predictor, updater, clutter_spatial_density=0.1, prob_detect=0.9)

    def run_pda_latest(oos_updater):
        # Only keep reference to the latest state
        state = prior
        for seconds_ in range(1, 11):
            hypotheses = hypothesiser.hypothesise(state, {detection(seconds_)}, time(seconds_))
            state = oos_updater.update(hypotheses)
            del hypotheses
            clear_caches()
        return state

    oos_updater = OOSUpdaterWrapper(updater, oos_predictor)
    oos_state = run_pda_latest(oos_updater)
    assert [seconds(past_state) for past_state in get_past_states(oos_state)] \
        == [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

    oos_updater = OOSUpdaterWrapper(
        updater, oos_predictor, min_history=datetime.timedelta(seconds=3))
    oos_state = run_pda_latest(oos_updater)
    assert [seconds(past_state) for past_state in get_past_states(oos_state)] == [10, 9, 8, 7]


@pytest.mark.parametrize('missed', [set(), {7.5}, {8}, {7, 9}], ids=str)
def test_min_history_out_of_sequence(oos_predictor, updater, predictor, prior, missed):
    oos_updater = OOSUpdaterWrapper(
        updater, oos_predictor, min_history=datetime.timedelta(seconds=3))

    # Out of sequence detection within min history
    order = [*range(1, 11), 7.5]
    oos_state = run(oos_predictor, oos_updater, prior, order, missed)
    expected_states = run_expected(predictor, updater, prior, order, missed)
    assert [seconds(past_state) for past_state in get_past_states(oos_state)] \
        == [10, 9, 8, 7.5, 7]
    assert_history_equal(oos_state, expected_states[:5])

    # Further out of sequence detection, after history reprocessed
    oos_state = run(oos_predictor, oos_updater, oos_state, [8.5])
    expected_states = run_expected(predictor, updater, prior, [*order, 8.5], missed)
    assert [seconds(past_state) for past_state in get_past_states(oos_state)] \
        == [10, 9, 8.5, 8, 7.5, 7]
    assert_history_equal(oos_state, expected_states[:6])


def test_min_history_exceeded(updater, predictor, prior):
    # No backward predictor
    oos_predictor = OOSPredictorWrapper(predictor)
    oos_updater = OOSUpdaterWrapper(
        updater, oos_predictor, min_history=datetime.timedelta(seconds=3))
    oos_state = run(oos_predictor, oos_updater, prior, range(1, 11))

    # History before min history has been released, so can't predict from it
    with pytest.raises(ValueError, match="Cannot predict"):
        oos_predictor.predict(oos_state, time(5.5))
