import copy
import datetime

import pytest
import numpy as np
from scipy.stats import multivariate_normal

try:
    import rtree
except (ImportError, AttributeError, OSError):
    # AttributeError or OSError raised when libspatialindex missing or unable to load.
    rtree = None

from ..neighbour import (
    NearestNeighbour, GlobalNearestNeighbour, GNNWith2DAssignment)
from ..probability import PDA, JPDA
from ..tree import DetectionKDTreeMixIn, TPRTreeMixIn
from ...models.measurement.nonlinear import CartesianToBearingRange
from ...predictor.particle import ParticlePredictor
from ...types.array import CovarianceMatrix, StateVectors
from ...types.detection import Detection, MissedDetection
from ...types.state import GaussianState, ParticleState
from ...types.track import Track
from ...updater.particle import ParticleUpdater


class DetectionKDTreeNN(DetectionKDTreeMixIn, NearestNeighbour):
    """DetectionKDTreeNN from NearestNeighbour and DetectionKDTreeMixIn"""
    pass


class DetectionKDTreeGNN(DetectionKDTreeMixIn, GlobalNearestNeighbour):
    """DetectionKDTreeGNN from GlobalNearestNeighbour and DetectionKDTreeMixIn"""
    pass


class DetectionKDTreeGNN2D(DetectionKDTreeMixIn, GNNWith2DAssignment):
    """DetectionKDTreeGNN2D from GNNWith2DAssignment and DetectionKDTreeMixIn"""
    pass


class TPRTreeNN(TPRTreeMixIn, NearestNeighbour):
    """TPRTreeNN from NearestNeighbour and TPRTreeMixIn"""
    pass


class TPRTreeGNN(TPRTreeMixIn, GlobalNearestNeighbour):
    """TPRTreeGNN from GlobalNearestNeighbour and TPRTreeMixIn"""
    pass


class TPRTreeGNN2D(TPRTreeMixIn, GNNWith2DAssignment):
    """TPRTreeGNN2D from GNNWith2DAssignment and TPRTreeMixIn"""
    pass


class KDTreePDA(DetectionKDTreeMixIn, PDA):
    pass


class KDTreeJPDA(DetectionKDTreeMixIn, JPDA):
    pass


class TPRTreePDA(TPRTreeMixIn, PDA):
    pass


class TPRTreeJPDA(TPRTreeMixIn, JPDA):
    pass


@pytest.fixture(params=[None, 1, 10])
def number_of_neighbours(request):
    return request.param


@pytest.fixture(params=[None, [1, 3]])
def vel_mapping(request):
    return request.param


@pytest.fixture(params=[
    DetectionKDTreeNN, DetectionKDTreeGNN, DetectionKDTreeGNN2D,
    TPRTreeNN, TPRTreeGNN, TPRTreeGNN2D])
def nn_associator(request, distance_hypothesiser, predictor,
                  updater, measurement_model, number_of_neighbours, vel_mapping):
    '''Distance associator for each KD Tree'''
    if 'KDTree' in request.param.__name__:
        return request.param(distance_hypothesiser, predictor,
                             updater, number_of_neighbours=number_of_neighbours)
    else:
        if rtree is None:
            return pytest.skip("'rtree' module not available")
        return request.param(distance_hypothesiser, measurement_model,
                             datetime.timedelta(hours=1), vel_mapping=vel_mapping)


@pytest.fixture(params=[KDTreePDA, KDTreeJPDA, TPRTreePDA, TPRTreeJPDA])
def pda_associator(request, probability_hypothesiser, predictor,
                   updater, measurement_model, number_of_neighbours, vel_mapping):
    if 'KDTree' in request.param.__name__:
        return request.param(probability_hypothesiser, predictor,
                             updater, number_of_neighbours=number_of_neighbours)
    else:
        if rtree is None:
            return pytest.skip("'rtree' module not available")
        return request.param(probability_hypothesiser, measurement_model,
                             datetime.timedelta(hours=1), vel_mapping=vel_mapping)


@pytest.fixture(params=[DetectionKDTreeGNN2D])
def probability_associator(request, probability_hypothesiser, predictor,
                           updater, measurement_model):
    '''Probability associator for each KD Tree'''
    return request.param(probability_hypothesiser, predictor, updater)


def test_nearest_neighbour(nn_associator):
    '''Test method for nearest neighbour and KD tree'''
    timestamp = datetime.datetime.now()
    t1 = Track([GaussianState(np.array([[0, 0, 0, 0]]), np.diag([1, 0.1, 1, 0.1]), timestamp)])
    t2 = Track([GaussianState(np.array([[3, 0, 3, 0]]), np.diag([1, 0.1, 1, 0.1]), timestamp)])
    d1 = Detection(np.array([[2, 2]]), timestamp)
    d2 = Detection(np.array([[5, 5]]), timestamp)

    tracks = {t1, t2}
    detections = {d1, d2}

    associations = nn_associator.associate(tracks, detections, timestamp)

    # There should be 2 associations
    assert len(associations) == 2

    # Each track should associate with a unique detection
    associated_measurements = [hypothesis.measurement
                               for hypothesis in associations.values()
                               if hypothesis.measurement]
    assert len(associated_measurements) == len(set(associated_measurements))
    if getattr(nn_associator, 'number_of_neighbours', None) is not None:
        assert len(associated_measurements) <= nn_associator.number_of_neighbours

    tracks = {}
    associations = nn_associator.associate(tracks, detections, timestamp)
    assert len(associations) == 0

    tracks = {t1, t2}
    detections = {}
    associations = nn_associator.associate(tracks, detections, timestamp)

    assert len([hypothesis for hypothesis in associations.values() if not hypothesis]) == 2


@pytest.mark.skipif(rtree is None, reason="'rtree' module not available")
def test_tpr_tree_management(distance_hypothesiser, measurement_model, vel_mapping, updater):
    '''Test method for TPR insert, delete and update'''
    nn_associator = TPRTreeNN(distance_hypothesiser, measurement_model,
                              datetime.timedelta(hours=1), vel_mapping=vel_mapping)
    timestamp = datetime.datetime.now()

    t1 = Track([GaussianState(np.array([[0, 0, 0, 0]]), np.diag([1, 0.1, 1, 0.1]), timestamp)])
    t2 = Track([GaussianState(np.array([[3, 0, 3, 0]]), np.diag([1, 0.1, 1, 0.1]), timestamp)])
    tracks = {t1, t2}

    d1 = Detection(np.array([[2, 2]]), timestamp)
    d2 = Detection(np.array([[5, 5]]), timestamp)

    detections = {d1, d2}

    # Insert
    # Run the associator to insert tracks in TPR tree and generate Updated tracks
    associations = nn_associator.associate(tracks, detections, timestamp)

    assert len(associations) == 2

    # Each track should associate with a unique detection
    associated_measurements = [hypothesis.measurement
                               for hypothesis in associations.values()
                               if hypothesis.measurement]
    assert len(associated_measurements) == len(set(associated_measurements))
    if getattr(nn_associator, 'number_of_neighbours', None) is not None:
        assert len(associated_measurements) <= nn_associator.number_of_neighbours

    for track, hypothesis in associations.items():
        if hypothesis:
            track.append(updater.update(hypothesis))
        else:
            track.append(hypothesis.prediction)

    # Update
    # At least one track has an updated state,
    # meaning we should hit the TPR tree update sub-routine
    timestamp = timestamp + datetime.timedelta(seconds=1)
    detections = {}
    associations = nn_associator.associate(tracks, detections, timestamp)

    assert len([hypothesis for hypothesis in associations.values() if not hypothesis]) == 2

    # Delete
    # Removing the second time should result in hitting the TPR tree deletion sub-routine
    tracks = {tracks.pop()}
    timestamp = timestamp + datetime.timedelta(seconds=1)
    associations = nn_associator.associate(tracks, detections, timestamp)

    assert len([hypothesis for hypothesis in associations.values() if not hypothesis]) == 1


@pytest.mark.skipif(rtree is None, reason="'rtree' module not available")
def test_tpr_tree_measurement_models(
        distance_hypothesiser, measurement_model, vel_mapping, updater):
    '''Test method for TPR insert, delete and update using non linear measurement model'''
    timestamp = datetime.datetime.now()
    measurement_model_nl = CartesianToBearingRange(
        ndim_state=4, mapping=[0, 2],
        noise_covar=CovarianceMatrix(np.diag([np.pi/180.0, 1])))
    nn_associator = TPRTreeNN(distance_hypothesiser, measurement_model,
                              datetime.timedelta(hours=1), vel_mapping=vel_mapping)

    t1 = Track([GaussianState(np.array([[0, 0, 0, 0]]), np.diag([1, 0.1, 1, 0.1]), timestamp)])
    t2 = Track([GaussianState(np.array([[3, 0, 3, 0]]), np.diag([1, 0.1, 1, 0.1]), timestamp)])
    tracks = {t1, t2}

    d1 = Detection(np.array([[2, 2]]), timestamp)
    d2 = Detection(np.array([[0.7854, 7.0711]]), timestamp, measurement_model=measurement_model_nl)

    detections = {d1, d2}

    # Insert
    # Run the associator to insert tracks in TPR tree and generate Updated tracks
    associations = nn_associator.associate(tracks, detections, timestamp)

    assert len(associations) == 2

    # Each track should associate with a unique detection
    associated_measurements = [hypothesis.measurement
                               for hypothesis in associations.values()
                               if hypothesis.measurement]
    assert len(associated_measurements) == len(set(associated_measurements))
    if getattr(nn_associator, 'number_of_neighbours', None) is not None:
        assert len(associated_measurements) <= nn_associator.number_of_neighbours


def test_missed_detection_nearest_neighbour(nn_associator):
    '''Test method for nearest neighbour and KD tree'''
    timestamp = datetime.datetime.now()
    t1 = Track([GaussianState(np.array([[0, 0, 0, 0]]), np.diag([1, 0.1, 1, 0.1]), timestamp)])
    t2 = Track([GaussianState(np.array([[3, 0, 3, 0]]), np.diag([1, 0.1, 1, 0.1]), timestamp)])
    #  extend to include to velocity !!!
    d1 = Detection(np.array([[20, 20]]), timestamp)

    tracks = {t1, t2}
    detections = {d1}

    associations = nn_associator.associate(tracks, detections, timestamp)

    # Best hypothesis should be missed detection hypothesis
    assert all(not hypothesis.measurement
               for hypothesis in associations.values())


def test_probability_gnn(probability_associator):
    '''Test method for global nearest neighbour and KD tree'''
    timestamp = datetime.datetime.now()
    t1 = Track([GaussianState(np.array([[0, 0, 0, 0]]), np.diag([1, 0.1, 1, 0.1]), timestamp)])
    t2 = Track([GaussianState(np.array([[3, 0, 3, 0]]), np.diag([1, 0.1, 1, 0.1]), timestamp)])
    d1 = Detection(np.array([[2, 2]]), timestamp)
    d2 = Detection(np.array([[5, 5]]), timestamp)

    tracks = {t1, t2}
    detections = {d1, d2}

    associations = probability_associator.associate(
        tracks, detections, timestamp)

    # There should be 2 associations
    assert len(associations) == 2

    # Each track should associate with a unique detection
    associated_measurements = [hypothesis.measurement
                               for hypothesis in associations.values()
                               if hypothesis.measurement]
    assert len(associated_measurements) == len(set(associated_measurements))
    if getattr(probability_associator, 'number_of_neighbours', None) is not None:
        assert len(associated_measurements) <= nn_associator.number_of_neighbours


def test_probability(pda_associator):

    timestamp = datetime.datetime.now()
    t1 = Track([GaussianState(np.array([[0, 0, 0, 0]]), np.diag([1, 0.1, 1, 0.1]), timestamp)])
    t2 = Track([GaussianState(np.array([[3, 0, 3, 0]]), np.diag([1, 0.1, 1, 0.1]), timestamp)])
    d1 = Detection(np.array([[0, 0]]), timestamp)
    d2 = Detection(np.array([[3, 3]]), timestamp)

    tracks = {t1, t2}
    detections = {d1, d2}

    associations = pda_associator.associate(tracks, detections, timestamp)

    # There should be 2 associations
    assert len(associations) == 2

    # verify association probabilities are correct
    prob_t1_d1_association = [hyp.probability for hyp in associations[t1]
                              if hyp.measurement is d1]
    prob_t1_d2_association = [hyp.probability for hyp in associations[t1]
                              if hyp.measurement is d2]
    prob_t2_d1_association = [hyp.probability for hyp in associations[t2]
                              if hyp.measurement is d1]
    prob_t2_d2_association = [hyp.probability for hyp in associations[t2]
                              if hyp.measurement is d2]
    number_of_neighbours = getattr(pda_associator, 'number_of_neighbours', None)
    if number_of_neighbours is None or number_of_neighbours > 1:
        assert prob_t1_d1_association[0] > prob_t1_d2_association[0]
        assert prob_t2_d1_association[0] < prob_t2_d2_association[0]
    else:
        assert prob_t1_d1_association
        assert prob_t2_d2_association
        assert not prob_t2_d1_association
        assert not prob_t1_d2_association


def test_missed_detection_probability(pda_associator):

    timestamp = datetime.datetime.now()
    t1 = Track([GaussianState(np.array([[0, 0, 0, 0]]), np.diag([1, 0.1, 1, 0.1]), timestamp)])
    t2 = Track([GaussianState(np.array([[3, 0, 3, 0]]), np.diag([1, 0.1, 1, 0.1]), timestamp)])
    d1 = Detection(np.array([[20, 20]]), timestamp)

    tracks = {t1, t2}
    detections = {d1}

    associations = pda_associator.associate(tracks, detections, timestamp)

    # Best hypothesis should be missed detection hypothesis
    max_track1_prob = max([hyp.probability for hyp in associations[t1]])
    max_track2_prob = max([hyp.probability for hyp in associations[t1]])

    track1_missed_detect_prob = max(
        [hyp.probability for hyp in associations[t1]
         if isinstance(hyp.measurement, MissedDetection)])
    track2_missed_detect_prob = max(
        [hyp.probability for hyp in associations[t1]
         if isinstance(hyp.measurement, MissedDetection)])

    assert max_track1_prob == track1_missed_detect_prob
    assert max_track2_prob == track2_missed_detect_prob


def test_no_detections_probability(pda_associator):

    timestamp = datetime.datetime.now()
    t1 = Track([GaussianState(np.array([[0, 0, 0, 0]]), np.diag([1, 0.1, 1, 0.1]), timestamp)])
    t2 = Track([GaussianState(np.array([[3, 0, 3, 0]]), np.diag([1, 0.1, 1, 0.1]), timestamp)])

    tracks = {t1, t2}
    detections = {}

    associations = pda_associator.associate(tracks, detections, timestamp)

    # All hypotheses should be missed detection hypothesis
    assert all(isinstance(hypothesis.measurement, MissedDetection)
               for multihyp in associations.values()
               for hypothesis in multihyp)


def test_no_tracks_probability(pda_associator):

    timestamp = datetime.datetime.now()
    d1 = Detection(np.array([[2, 2]]), timestamp)
    d2 = Detection(np.array([[5, 5]]), timestamp)

    tracks = {}
    detections = {d1, d2}

    associations = pda_associator.associate(tracks, detections, timestamp)

    # Since no Tracks went in, there should be no associations
    assert not associations


def test_particle_tree(nn_associator):
    timestamp = datetime.datetime.now()
    p1 = multivariate_normal.rvs(np.array([0, 0, 0, 0]),
                                 np.diag([1, 0.1, 1, 0.1]),
                                 size=200)
    p2 = multivariate_normal.rvs(np.array([3, 0, 3, 0]),
                                 np.diag([1, 0.1, 1, 0.1]),
                                 size=200)
    t1 = Track([ParticleState(StateVectors(p1.T), timestamp, np.full(200, 1 / 200))])
    t2 = Track([ParticleState(StateVectors(p2.T), timestamp, np.full(200, 1 / 200))])
    d1 = Detection(np.array([[2, 2]]), timestamp)
    d2 = Detection(np.array([[5, 5]]), timestamp)

    tracks = {t1, t2}
    detections = {d1, d2}

    # Switch predictor/updater to Particle ones.
    nn_associator.hypothesiser.predictor = ParticlePredictor(
        nn_associator.hypothesiser.predictor.transition_model)
    nn_associator.hypothesiser.updater = ParticleUpdater(
        nn_associator.hypothesiser.updater.measurement_model)
    if isinstance(nn_associator, DetectionKDTreeMixIn):
        nn_associator.predictor = nn_associator.hypothesiser.predictor
        nn_associator.updater = nn_associator.hypothesiser.updater
    associations = nn_associator.associate(tracks, detections, timestamp)

    # There should be 2 associations
    assert len(associations) == 2

    # Each track should associate with a unique detection
    associated_measurements = [hypothesis.measurement
                               for hypothesis in associations.values()
                               if hypothesis.measurement]
    assert len(associated_measurements) == len(set(associated_measurements))
    if getattr(nn_associator, 'number_of_neighbours', None) is not None:
        assert len(associated_measurements) <= nn_associator.number_of_neighbours

    tracks = {}
    associations = nn_associator.associate(tracks, detections, timestamp)
    assert len(associations) == 0

    tracks = {t1, t2}
    detections = {}
    associations = nn_associator.associate(tracks, detections, timestamp)

    assert len([hypothesis for hypothesis in associations.values() if not hypothesis]) == 2


def test_kd_tree_measurement_models(distance_hypothesiser, predictor, updater, measurement_model):
    timestamp = datetime.datetime.now()

    t1 = Track([GaussianState(np.array([[0, 0, 0, 0]]), np.diag([1, 0.1, 1, 0.1]), timestamp)])
    t2 = Track([GaussianState(np.array([[3, 0, 3, 0]]), np.diag([1, 0.1, 1, 0.1]), timestamp)])
    tracks = {t1, t2}

    d1 = Detection(np.array([[2, 0]]), measurement_model=measurement_model, timestamp=timestamp)
    d2 = Detection(np.array([[5, 0]]), measurement_model=measurement_model, timestamp=timestamp)
    detections = {d1, d2}

    updater = copy.copy(updater)
    updater.measurement_model = None  # Must therefore use model on detections
    nn_associator = DetectionKDTreeNN(distance_hypothesiser, predictor, updater, max_distance=10)
    associations = nn_associator.associate(tracks, detections, timestamp)

    # There should be 2 associations
    assert len(associations) == 2

    measurement_model = copy.copy(measurement_model)
    measurement_model.mapping = [0, 1]

    d2.measurement_model = measurement_model  # Model mismatch from d1

    with pytest.raises(
            RuntimeError, match="KDTree requires all detections have same measurement model"):
        nn_associator.associate(tracks, detections, timestamp)
