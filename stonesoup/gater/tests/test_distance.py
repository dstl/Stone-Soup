import datetime

import numpy as np
import pytest

from ..distance import DistanceGater, TrackingStateSpaceDistanceGater
from ... import measures as measures
from ...hypothesiser.distance import DistanceHypothesiser
from ...hypothesiser.probability import PDAHypothesiser
from ...models.measurement.linear import LinearGaussian
from ...models.measurement.nonlinear import CartesianToBearingRange, Cartesian2DToBearing, \
    CombinedReversibleGaussianMeasurementModel
from ...models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity
from ...predictor.kalman import ExtendedKalmanPredictor
from ...types.detection import Detection
from ...types.hypothesis import SingleHypothesis
from ...types.state import GaussianState
from ...types.track import Track
from ...types.update import GaussianStateUpdate
from ...updater.kalman import ExtendedKalmanUpdater

measure = measures.Mahalanobis()


@pytest.mark.parametrize(
    "detections, gate_threshold, num_gated",
    [
        (   # Test 1
            {Detection(np.array([[2]])), Detection(np.array([[3]])), Detection(np.array([[6]])),
             Detection(np.array([[0]])), Detection(np.array([[-1]])), Detection(np.array([[-4]]))},
            1,
            3
        ),
        (   # Test 2
            {Detection(np.array([[2]])), Detection(np.array([[3]])), Detection(np.array([[6]])),
             Detection(np.array([[0]])), Detection(np.array([[-1]])), Detection(np.array([[-4]]))},
            2,
            5
        ),
        (   # Test 3
            {Detection(np.array([[2]])), Detection(np.array([[3]])), Detection(np.array([[6]])),
             Detection(np.array([[0]])), Detection(np.array([[-1]])), Detection(np.array([[-4]]))},
            4,
            7
        )
    ],
    ids=["test1", "test2", "test3"]
)
def test_distance(predictor, updater, detections, gate_threshold, num_gated):

    timestamp = datetime.datetime.now()

    hypothesiser = PDAHypothesiser(
        predictor, updater, clutter_spatial_density=0.000001, include_all=True)
    gater = DistanceGater(hypothesiser, measure=measure, gate_threshold=gate_threshold)

    track = Track([GaussianStateUpdate(
                    np.array([[0]]),
                    np.array([[1]]),
                    SingleHypothesis(
                        None,
                        Detection(np.array([[0]]), metadata={"MMSI": 12345})),
                    timestamp=timestamp)])

    hypotheses = gater.hypothesise(track, detections, timestamp)

    # The number of gated hypotheses matches the expected
    assert len(hypotheses) == num_gated

    # The gated hypotheses are either the null hypothesis or their distance is less than the set
    # gate threshold
    assert all(not hypothesis.measurement or
               measure(hypothesis.measurement_prediction, hypothesis.measurement) < gate_threshold
               for hypothesis in hypotheses)

    # There is a SINGLE missed detection hypothesis
    assert len([hypothesis for hypothesis in hypotheses if not hypothesis]) == 1


@pytest.fixture()
def standard_distance_hypothesiser():
    transition_model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.05),
                                                              ConstantVelocity(0.05)])
    predictor = ExtendedKalmanPredictor(transition_model)
    updater = ExtendedKalmanUpdater(bearing_range_measurement_model)
    hypothesiser = DistanceHypothesiser(predictor, updater, measure=measures.Euclidean())
    return hypothesiser


@pytest.fixture()
def hypothesiser_with_tracking_state_space_distance_gater(standard_distance_hypothesiser):
    hypothesiser_with_gate = TrackingStateSpaceDistanceGater(
        hypothesiser=standard_distance_hypothesiser,
        measure=measures.Euclidean(),
        gate_threshold=10,
    )
    return hypothesiser_with_gate


start_time = datetime.datetime(2024, 1, 1)

track = Track([GaussianState(
    state_vector=[0, 0, 100, 0],
    timestamp=start_time,
    covar=np.diag([1.5, 0.5, 1.5, 0.5])
)])

bearing_range_measurement_model = CartesianToBearingRange(
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.diag([np.radians(0.2), 1]),  # Covariance matrix. 0.2 degree variance in
    # bearing and 1 metre in range
    translation_offset=np.array([[0], [0]])  # Offset measurements to location of
    # sensor in cartesian.
)

linear_measurement_model = LinearGaussian(
    ndim_state=4,
    mapping=(0, 2),
    noise_covar=np.diag([1, 1]),  # Covariance matrix. 1 each in x and y
)


@pytest.mark.parametrize(
    "detection",
    [
        # Perfect detection
        Detection(state_vector=[np.pi/2, 100],
                  timestamp=start_time,
                  measurement_model=bearing_range_measurement_model),
        # Detection with slightly wrong range
        Detection(state_vector=[np.pi/2, 109],
                  timestamp=start_time,
                  measurement_model=bearing_range_measurement_model),
        # Detection with slightly wrong bearing
        Detection(state_vector=[0.01 + np.pi/2, 109],
                  timestamp=start_time,
                  measurement_model=bearing_range_measurement_model),
        # Perfect Linear Detection
        Detection(state_vector=[0, 100],
                  timestamp=start_time,
                  measurement_model=linear_measurement_model),
    ],
    ids=["bearing_range_1", "bearing_range_2", "bearing_range_3", "linear_4"]
)
def test_tracking_state_space_distance_gater_good_detections(
        detection, standard_distance_hypothesiser,
        hypothesiser_with_tracking_state_space_distance_gater):

    hypothesiser_with_gate = hypothesiser_with_tracking_state_space_distance_gater

    hypotheses_no_gate = standard_distance_hypothesiser.hypothesise(
        track, {detection}, timestamp=start_time)
    hypotheses_with_gate = hypothesiser_with_gate.hypothesise(
        track, {detection}, timestamp=start_time)

    assert len(hypotheses_no_gate) == len(hypotheses_with_gate) == 2


@pytest.mark.parametrize(
    "detection",
    [
        # Correct range. Wrong bearing
        Detection(state_vector=[0, 100],
                  timestamp=start_time,
                  measurement_model=bearing_range_measurement_model),
        # Correct bearing. Wrong range
        Detection(state_vector=[np.pi / 2, 111],
                  timestamp=start_time,
                  measurement_model=bearing_range_measurement_model),
        # Detection with slightly wrong bearing
        Detection(state_vector=[-np.pi / 2, 109],
                  timestamp=start_time,
                  measurement_model=bearing_range_measurement_model),
        # Wrong Linear Detection
        Detection(state_vector=[10, 110],
                  timestamp=start_time,
                  measurement_model=linear_measurement_model),
    ],
    ids=["bearing_range_1", "bearing_range_2", "bearing_range_3", "linear_4"]
)
def test_tracking_state_space_distance_gater_bad_detections(
        detection, standard_distance_hypothesiser,
        hypothesiser_with_tracking_state_space_distance_gater):

    hypothesiser_with_gate = hypothesiser_with_tracking_state_space_distance_gater

    hypotheses_no_gate = standard_distance_hypothesiser.hypothesise(
        track, {detection}, timestamp=start_time)
    hypotheses_with_gate = hypothesiser_with_gate.hypothesise(
        track, {detection}, timestamp=start_time)

    assert len(hypotheses_no_gate) == 2
    assert len(hypotheses_with_gate) == 1


def test_tracking_state_space_distance_gater_value_error(
        hypothesiser_with_tracking_state_space_distance_gater):

    detection = Detection(state_vector=[0, 100], timestamp=start_time, measurement_model=None)

    hypothesiser_with_gate = hypothesiser_with_tracking_state_space_distance_gater

    with pytest.raises(ValueError):
        hypothesiser_with_gate.hypothesise(track, {detection}, timestamp=start_time)


bearing_measurement_model = Cartesian2DToBearing(ndim_state=4,
                                                 mapping=(0, 2),
                                                 noise_covar=np.diag([1]))


@pytest.mark.parametrize(
    "detection",
    [
        # Irreversible detection
        Detection(state_vector=[0, 100],
                  timestamp=start_time,
                  measurement_model=bearing_measurement_model
                  ),
        # Irreversible detection 2
        Detection(state_vector=[0, 100],
                  timestamp=start_time,
                  measurement_model=CombinedReversibleGaussianMeasurementModel(
                      [bearing_measurement_model])
                  ),
    ],
    ids=["test_1", "test_2"]
)
@pytest.mark.parametrize("allow_non_reversible_detections", [True, False])
def test_tracking_state_space_distance_gater_irreversible_detections(
        detection, allow_non_reversible_detections,
        hypothesiser_with_tracking_state_space_distance_gater):

    hypothesiser_with_gate = hypothesiser_with_tracking_state_space_distance_gater
    hypothesiser_with_gate.allow_non_reversible_detections = allow_non_reversible_detections

    hypotheses_with_gate = hypothesiser_with_gate.hypothesise(
        track, {detection}, timestamp=start_time)

    if allow_non_reversible_detections:
        assert len(hypotheses_with_gate) == 2
    else:
        assert len(hypotheses_with_gate) == 1
