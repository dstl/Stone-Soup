from datetime import datetime, timedelta
import numpy as np

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

start_time = datetime.now().replace(second=0, microsecond=0)
np.random.seed(100)

number_of_targets = 2
range_value = 10000
max_segments = 3
max_segment_length = 125
min_segment_length = 60
max_disjoint_length = 250
min_disjoint_length = 125
max_track_start = 125
n_spacial_dimensions = 2
measurement_noise = 1000

truths = set()
truthlets = set()
tracklets = set()
all_measurements = []
all_tracks = set()

transition_model = CombinedLinearGaussianTransitionModel(
    [ConstantVelocity(1)]*n_spacial_dimensions, seed=12)
measurement_cov_array = np.zeros((n_spacial_dimensions, n_spacial_dimensions), int)
np.fill_diagonal(measurement_cov_array, measurement_noise)
measurement_model = LinearGaussian(ndim_state=2*n_spacial_dimensions,
                                   mapping=list(range(0, 2*n_spacial_dimensions,
                                                      2)), noise_covar=measurement_cov_array)


def tracker(all_measurements, initiator, deleter, data_associator,
            hypothesiser, predictor, updater):
    tracks = set()
    historic_tracks = set()
    for n, measurements in enumerate(all_measurements):
        hypotheses = data_associator.associate(tracks, measurements,
                                               start_time + timedelta(seconds=n))
        associated_measurements = set()
        for track in tracks:
            hypothesis = hypotheses[track]
            if hypothesis.measurement:
                post = updater.update(hypothesis)
                track.append(post)
                associated_measurements.add(hypothesis.measurement)
            else:
                track.append(hypothesis.prediction)
        del_tracks = deleter.delete_tracks(tracks)
        tracks -= del_tracks
        tracks |= initiator.initiate(measurements - associated_measurements,
                                     start_time + timedelta(seconds=n))
        historic_tracks |= del_tracks
    historic_tracks |= tracks
    return historic_tracks


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
                       hypothesiser, predictor, updater)
    for t in tracklet:
        all_tracks.add(t)

transition_model = CombinedLinearGaussianTransitionModel(
    [OrnsteinUhlenbeck(0.001, 2e-2)]*n_spacial_dimensions, seed=12)

predictor = KalmanPredictor(transition_model)
hypothesiser = DistanceHypothesiser(predictor, updater, Mahalanobis(), missed_distance=300)
stitcher = TrackStitcher(forward_hypothesiser=hypothesiser)

stitched_tracks = stitcher.stitch(all_tracks, start_time)


def test_correct_no_stitched_tracks():
    no_stitched_tracks = len(stitched_tracks)
    assert no_stitched_tracks == 2
