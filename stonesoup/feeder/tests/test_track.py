# -*- coding: utf-8 -*-
import pytest
import numpy as np

from ...types.state import GaussianState
from ...types.track import Track
from ..track import Tracks2GaussianDetectionFeeder

t1 = Track(GaussianState([1, 1, 1, 1], np.diag([2, 2, 2, 2]), timestamp=2))
t2 = Track([GaussianState([1, 1, 1, 1], np.diag([2, 2, 2, 2]), timestamp=1),
            GaussianState([2, 1, 2, 1], np.diag([2, 2, 2, 2]), timestamp=2)])
t3 = Track([GaussianState([1, 1], np.diag([2, 2]), timestamp=0),
            GaussianState([2, 1], np.diag([2, 2]), timestamp=1),
            GaussianState([3, 1], np.diag([2, 2]), timestamp=2)])
t4 = Track(GaussianState([1, 0, 1, 0, 1, 0], np.diag([2, 2, 2, 2, 2, 2]), timestamp=2))


@pytest.mark.parametrize(
    "tracks",
    [
        ([t1]),
        ([t1, t2]),
        ([t2, t3]),
        ([t1, t2, t3, t4])
    ]
)
def test_Track2GaussianDetectionFeeder(tracks):

    # Make feeder and get detections
    reader = [(tracks[0][-1].timestamp, tracks)]
    feeder = Tracks2GaussianDetectionFeeder(reader=reader)
    time, detections = next(feeder.data_gen())

    # Check that there are the right number of detections
    assert len(detections) == len(tracks)

    # Check that the correct state was taken from each track
    assert np.all([detections[i].timestamp == tracks[i][-1].timestamp for i in range(len(tracks))])
    assert np.all([detections[i].timestamp == time for i in range(len(tracks))])

    # Check that the dimension of each detection is correct
    assert np.all([len(detections[i].state_vector) == len(tracks[i][-1].state_vector)
                   for i in range(len(tracks))])
    assert np.all([len(detection.state_vector) == detection.measurement_model.ndim
                   for detection in detections])

    # Check that the detection has the correct mean and covariance
    for i in range(len(tracks)):
        assert np.all(detections[i].state_vector == tracks[i][-1].state_vector)
        assert np.all(detections[i].covar == tracks[i][-1].covar)
