# -*- coding: utf-8 -*-
import datetime

import pytest
import numpy as np

from ..neighbour import (NearestNeighbour, GlobalNearestNeighbour)
from ...types.track import Track
from ...types.detection import Detection
from ...types.state import GaussianState


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
    associated_measurements = [hypothesis.measurement
                               for hypothesis in associations.values()
                               if hypothesis.measurement]
    assert len(associated_measurements) == len(set(associated_measurements))
