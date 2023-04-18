from datetime import datetime, timedelta
import numpy as np
import pytest

from stonesoup.types.state import GaussianState
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, \
    ConstantVelocity, OrnsteinUhlenbeck
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.types.detection import TrueDetection
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.deleter.multi import CompositeDeleter
from stonesoup.deleter.time import UpdateTimeStepsDeleter
from stonesoup.initiator.simple import MultiMeasurementInitiator
from stonesoup.types.groundtruth import GroundTruthPath, GroundTruthState
from stonesoup.stitcher import TrackStitcher

@pytest.fixture
def params():
    start_time = datetime(2023, 4, 6, 16, 17)
    np.random.seed(100)

    number_of_targets = 2
    range_value = 10000
    max_segments = 5
    max_segment_length = 125
    min_segment_length = 120
    max_disjoint_length = 125
    min_disjoint_length = 120
    max_track_start = 125
    n_spacial_dimensions = 3
    measurement_noise = 1000

    truths = set()
    truthlets = set()
    tracklets = set()
    all_measurements = []
    all_tracks = set()

    transition_model = CombinedLinearGaussianTransitionModel(
        [ConstantVelocity(1)]*n_spacial_dimensions, seed=435)
    measurement_cov_array = np.zeros((n_spacial_dimensions, n_spacial_dimensions), int)
    np.fill_diagonal(measurement_cov_array, measurement_noise)
    measurement_model = LinearGaussian(ndim_state=2*n_spacial_dimensions,
                                       mapping=list(range(0, 2*n_spacial_dimensions,
                                                          2)), noise_covar=measurement_cov_array, seed=435)

    from stonesoup.stitcher import tracker

    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model)
    hypothesiser = DistanceHypothesiser(predictor, updater, Mahalanobis(), missed_distance=30)
    data_associator = GNNWith2DAssignment(hypothesiser)
    deleter = CompositeDeleter([UpdateTimeStepsDeleter(50), CovarianceBasedDeleter(5000)])
    initiator = MultiMeasurementInitiator(prior_state=GaussianState(
        np.zeros((2*n_spacial_dimensions, 1), int), np.diag([1, 0]*n_spacial_dimensions)),
        measurement_model=measurement_model, deleter=deleter, data_associator=data_associator,
        updater=updater, min_points=2)
    state_vector = [np.random.uniform(-range_value, range_value, 1),
                    np.random.uniform(-2, 2, 1)]*n_spacial_dimensions

    for i in range(number_of_targets):
        number_of_segments = int(np.random.choice(range(1, max_segments), 1))
        truthlet0_length = np.random.choice(range(max_track_start), 1)
        truthlet_lengths = np.random.choice(range(min_segment_length, max_segment_length),
                                            number_of_segments)
        disjoint_lengths = np.random.choice(range(min_disjoint_length, max_disjoint_length),
                                            number_of_segments)
        segment_pair_lengths = np.insert(truthlet_lengths + disjoint_lengths, 0, truthlet0_length,
                                         axis=0)
        truthlet_startpoints = np.cumsum(segment_pair_lengths)
        truthlet_endpoints = truthlet_startpoints + np.append(truthlet_lengths, 0)
        starts = truthlet_startpoints[:number_of_segments]
        stops = truthlet_endpoints[:number_of_segments]
        truth = GroundTruthPath([GroundTruthState(state_vector, timestamp=start_time)],
                                id=i)
        for k in range(1, np.max(stops)):
            truth.append(GroundTruthState(
                transition_model.function(truth[k-1], noise=True, time_interval=timedelta(seconds=1)),
                timestamp=truth[k-1].timestamp + timedelta(seconds=1)))
        for j in range(number_of_segments):
            truthlet = GroundTruthPath(truth[starts[j]:stops[j]],
                                       id=str("G::" + str(truth.id) + "::S::" + str(j) + "::"))
            truthlets.add(truthlet)
        truths.add(truth)

    for n, truthlet in enumerate(truthlets):
        measurementlet = []
        for state in truthlet:
            m = measurement_model.function(state, noise=True)
            m0 = TrueDetection(m,
                               timestamp=state.timestamp,
                               measurement_model=measurement_model,
                               groundtruth_path=truthlet)
            measurementlet.append({m0})
            all_measurements.append({m0})
        tracklet = tracker(measurementlet, initiator, deleter, data_associator,
                           hypothesiser, predictor, updater, start_time)
        for t in tracklet:
            all_tracks.add(t)

    transition_model = CombinedLinearGaussianTransitionModel(
        [OrnsteinUhlenbeck(0.001, 2e-2)]*n_spacial_dimensions, seed=435)

    predictor = KalmanPredictor(transition_model)
    hypothesiser = DistanceHypothesiser(predictor, updater, Mahalanobis(), missed_distance=500)

    return {"hypothesiser": hypothesiser, "start": start_time, "all_tracks": all_tracks}


def test_correct_no_stitched_tracks(params):
    hypothesiser = params["hypothesiser"]
    all_tracks = params["all_tracks"]
    start_time = params["start"]
    stitcher = TrackStitcher(forward_hypothesiser=hypothesiser)
    stitched_tracks, _ = stitcher.stitch(all_tracks, start_time)
    no_stitched_tracks = len(stitched_tracks)
    assert no_stitched_tracks == 5


def test_correct_no_stitched_tracks1(params):
    print(type(params))
    hypothesiser = params["hypothesiser"]
    all_tracks = params["all_tracks"]
    start_time = params["start"]
    stitcher1 = TrackStitcher(backward_hypothesiser=hypothesiser)
    stitched_tracks1, _ = stitcher1.stitch(all_tracks, start_time)
    no_stitched_tracks = len(stitched_tracks1)
    assert no_stitched_tracks == 5


def test_correct_no_stitched_tracks2(params):
    hypothesiser = params["hypothesiser"]
    all_tracks = params["all_tracks"]
    start_time = params["start"]
    stitcher2 = TrackStitcher(forward_hypothesiser=hypothesiser,
                              backward_hypothesiser=hypothesiser)
    stitched_tracks2, _ = stitcher2.stitch(all_tracks, start_time)
    # merge_forward_and_backward() has a bug. It is not returning x.
    no_stitched_tracks = len(stitched_tracks2)
    assert no_stitched_tracks == 5
