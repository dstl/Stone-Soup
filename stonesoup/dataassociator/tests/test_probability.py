# -*- coding: utf-8 -*-
import datetime

import pytest
import numpy as np

from ..probability import SimplePDA
from ...types import Track, Detection, GaussianState, MissedDetection


@pytest.fixture(params=[SimplePDA])
def associator(request, proabability_hypothesiser):
    return request.param(proabability_hypothesiser)


def test_probability(associator):

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
                               if hypothesis.measurement is not None]
    assert len(associated_measurements) == len(set(associated_measurements))


def test_missed_detection_probability(associator):

    timestamp = datetime.datetime.now()
    t1 = Track([GaussianState(np.array([[0]]), np.array([[1]]), timestamp)])
    t2 = Track([GaussianState(np.array([[3]]), np.array([[1]]), timestamp)])
    d1 = Detection(np.array([[20]]))

    tracks = {t1, t2}
    detections = {d1}

    associations = associator.associate(tracks, detections, timestamp)

    # Best hypothesis should be missed detection hypothesis
    assert all(isinstance(hypothesis.measurement, MissedDetection)
               for hypothesis in associations.values())
