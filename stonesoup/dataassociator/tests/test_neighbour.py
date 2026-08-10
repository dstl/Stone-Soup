import datetime

import pytest
import numpy as np

from ..neighbour import (
    NearestNeighbour, GlobalNearestNeighbour, GNNWith2DAssignment)
from ...hypothesiser.distance import DistanceHypothesiser
from ...measures import Mahalanobis
from ...types.detection import Detection
from ...types.state import GaussianState
from ...types.track import Track


@pytest.fixture(params=[
    NearestNeighbour, GlobalNearestNeighbour, GNNWith2DAssignment])
def associator(request, distance_hypothesiser):
    return request.param(distance_hypothesiser)


@pytest.fixture(params=[GNNWith2DAssignment])
def probability_associator(request, probability_hypothesiser):
    return request.param(probability_hypothesiser)


def test_nearest_neighbour(associator):
    timestamp = datetime.datetime.now()
    t1 = Track([GaussianState(np.array([[0, 0, 0, 0]]), np.diag([1, 0.1, 1, 0.1]), timestamp)])
    t2 = Track([GaussianState(np.array([[3, 0, 3, 0]]), np.diag([1, 0.1, 1, 0.1]), timestamp)])
    d1 = Detection(np.array([[2, 2]]), timestamp)
    d2 = Detection(np.array([[5, 5]]), timestamp)

    tracks = {t1, t2}
    detections = {d1, d2}

    associations = associator.associate(tracks, detections, timestamp)

    # There should be 2 associations
    assert len(associations) == 2

    # Each track should associate with a unique detection
    associated_measurements = [hypothesis.measurement
                               for hypothesis in associations.values()
                               if hypothesis.measurement]
    assert len(associated_measurements) == len(set(associated_measurements))


def test_missed_detection_nearest_neighbour(associator):
    timestamp = datetime.datetime.now()
    t1 = Track([GaussianState(np.array([[0, 0, 0, 0]]), np.diag([1, 0.1, 1, 0.1]), timestamp)])
    t2 = Track([GaussianState(np.array([[3, 0, 3, 0]]), np.diag([1, 0.1, 1, 0.1]), timestamp)])
    d1 = Detection(np.array([[20, 20]]), timestamp)

    tracks = {t1, t2}
    detections = {d1}

    associations = associator.associate(tracks, detections, timestamp)

    # Best hypothesis should be missed detection hypothesis
    assert all(not hypothesis.measurement
               for hypothesis in associations.values())


@pytest.mark.parametrize(
    'associator_class', [NearestNeighbour, GlobalNearestNeighbour, GNNWith2DAssignment])
def test_infinite_missed_distance(associator_class, predictor, updater):
    # Default missed distance is infinite, such that every joint hypothesis has an
    # infinite distance where a missed detection is unavoidable: #824
    associator = associator_class(
        DistanceHypothesiser(predictor, updater, Mahalanobis()))

    timestamp = datetime.datetime.now()
    tracks = [
        Track([GaussianState(np.array([[x, 0, x, 0]]), np.diag([1, 0.1, 1, 0.1]), timestamp)])
        for x in (0, 3, 6)]
    detection = Detection(np.array([[6, 6]]), timestamp)

    # Tracks passed in order, nearest last, so that an arbitrary (first) choice is wrong
    associations = associator.associate(tracks, {detection}, timestamp)

    # Detection must be associated to the nearest (last) track, not an arbitrary one
    assert associations[tracks[-1]].measurement is detection
    assert not associations[tracks[0]]
    assert not associations[tracks[1]]


def test_probability_gnn(probability_associator):
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
