import pytest
import numpy as np

from ...types.state import GaussianState
from ...types.track import Track
from ..track import Tracks2GaussianDetectionFeeder, ReplayTrackFeeder

t1 = Track(GaussianState([1, 1, 1, 1], np.diag([2, 2, 2, 2]), timestamp=2))
t2 = Track([GaussianState([1, 1, 1, 1], np.diag([2, 2, 2, 2]), timestamp=1),
            GaussianState([2, 1, 2, 1], np.diag([2, 2, 2, 2]), timestamp=2)])
t3 = Track([GaussianState([1, 1], np.diag([2, 2]), timestamp=0),
            GaussianState([2, 1], np.diag([2, 2]), timestamp=1),
            GaussianState([3, 1], np.diag([2, 2]), timestamp=2)])
t4 = Track(GaussianState([1, 0, 1, 0, 1, 0], np.diag([2, 2, 2, 2, 2, 2]), timestamp=2))

times = [0, 1, 2]


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
    assert np.all([detection.timestamp == track.timestamp
                   for track in tracks
                   for detection in detections if detection.metadata['track_id'] == track.id])
    assert np.all([detection.timestamp == time for detection in detections])

    # Check that the dimension of each detection is correct
    assert np.all([len(detection.state_vector) == len(track.state_vector)
                   for track in tracks
                   for detection in detections if detection.metadata['track_id'] == track.id])
    assert np.all([len(detection.state_vector) == detection.measurement_model.ndim
                   for detection in detections])

    # Check that the detection has the correct mean and covariance
    for track in tracks:
        detection = next(iter(detection
                              for detection in detections
                              if detection.metadata['track_id'] == track.id))
        assert np.all(detection.state_vector == track.state_vector)
        assert np.all(detection.covar == track.covar)


@pytest.mark.parametrize(("tracks", "times"),
                         [([t1], times), ([t1], None),
                          ([t1, t2], times), ([t1, t2], None),
                          ([t2, t3], times), ([t2, t3], None),
                          ([t1, t2, t3, t4], times), ([t1, t2, t3, t4], None)])
def test_ReplayTrackFeeder(tracks, times):
    feeder = ReplayTrackFeeder(reader=tracks, times=times)
    feeder_times = []
    feeder_tracks = set()
    for new_time, new_tracks in feeder:
        feeder_times.append(new_time)
        feeder_tracks |= new_tracks
    if times is not None:
        assert times == feeder_times
    assert len(tracks) == len(feeder_tracks)
    assert (sorted([len(track) for track in tracks]) ==
            sorted([len(track) for track in feeder_tracks]))
