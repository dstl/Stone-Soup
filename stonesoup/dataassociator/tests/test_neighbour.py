# -*- coding: utf-8 -*-
import datetime

import pytest
import numpy as np

from ..neighbour import NearestNeighbour, GlobalNearestNeighbour
from ...types import Track, Detection, GaussianState


@pytest.fixture(params=[NearestNeighbour, GlobalNearestNeighbour])
def associator(request, hypothesiser):
    return request.param(hypothesiser)


def test_nearest_neighbour(associator):

    timestamp = datetime.datetime.now()
    t1 = Track([GaussianState(np.array([[0]]), np.array([[1]]), timestamp)])
    t2 = Track([GaussianState(np.array([[3]]), np.array([[1]]), timestamp)])
    d1 = Detection(np.array([[2]]))
    d2 = Detection(np.array([[5]]))

    tracks = {t1, t2}
    detections = {d1, d2}

    associations = associator.associate(tracks, detections, timestamp)

    # There should be 2 associations
    assert len(associations) == 2

    # Each track should associate with a unique detection
    associated_detections = [hypothesis.detection
                             for hypothesis in associations.values()
                             if hypothesis.detection is not None]
    assert len(associated_detections) == len(set(associated_detections))


def test_missed_detection_nearest_neighbour(associator):

    timestamp = datetime.datetime.now()
    t1 = Track([GaussianState(np.array([[0]]), np.array([[1]]), timestamp)])
    t2 = Track([GaussianState(np.array([[3]]), np.array([[1]]), timestamp)])
    d1 = Detection(np.array([[20]]))

    tracks = {t1, t2}
    detections = {d1}

    associations = associator.associate(tracks, detections, timestamp)

    # Best hypothesis should be missed detection hypothesis
    assert all(hypothesis.detection is None
               for hypothesis in associations.values())
