import datetime

import numpy as np
import pytest

from ..error import CovarianceBasedDeleter
from ..multi import CompositeDeleter
from ..time import UpdateTimeDeleter
from ...types.detection import Detection
from ...types.hypothesis import SingleHypothesis
from ...types.state import GaussianState
from ...types.track import Track
from ...types.update import GaussianStateUpdate


@pytest.fixture(params=[True, False])
def intersect(request):
    return request.param


def test_multi_deleter_single(intersect):
    """Test multi deleter classes with a single deleter"""

    # Create covariance based deleter
    timestamp = datetime.datetime.now()
    state = GaussianState(
        np.array([[0], [0]]),
        np.array([[100, 0], [0, 1]]), timestamp)
    track = Track()
    track.append(state)
    tracks = {track}
    state = GaussianState(
        np.array([[0], [0]]),
        np.array([[1, 0], [0, 1]]), timestamp)
    track = Track()
    track.append(state)
    tracks.add(track)
    covar_deletion_thresh = 100
    deleter = CovarianceBasedDeleter(covar_trace_thresh=covar_deletion_thresh)

    # Test intersect deleter
    multi_deleter = CompositeDeleter(deleters=[deleter], intersect=intersect)
    deleted_tracks = multi_deleter.delete_tracks(tracks)
    tracks -= deleted_tracks

    assert (len(tracks) == 1)
    assert (len(deleted_tracks) == 1)


def test_multi_deleter_multiple(intersect):
    """Test multi deleter classes with multiple deleters"""

    covar_deletion_thresh = 99
    deleter = CovarianceBasedDeleter(covar_trace_thresh=covar_deletion_thresh)
    deleter2 = UpdateTimeDeleter(datetime.timedelta(minutes=10))
    multi_deleter = CompositeDeleter(deleters=[deleter, deleter2], intersect=intersect)

    # Create track that is not deleted by either deleter
    track = Track([
        GaussianState(
            np.array([[0]]),
            np.array([[10]]),
            timestamp=datetime.datetime(2018, 1, 1, 10)),
        GaussianStateUpdate(
            [[0]],
            np.array([[10]]),
            SingleHypothesis(None, Detection([[0]])),
            timestamp=datetime.datetime(2018, 1, 1, 14))
    ])
    tracks = {track}
    deleted_tracks = multi_deleter.delete_tracks(tracks)
    tracks -= deleted_tracks

    assert (len(tracks) == 1)
    assert (len(deleted_tracks) == 0)

    # Create track that is deleted by cbd but not time deleter
    track = Track([
        GaussianState(
            np.array([[0]]),
            np.array([[100]]),
            timestamp=datetime.datetime(2018, 1, 1, 10)),
        GaussianStateUpdate(
            [[0]],
            np.array([[100]]),
            SingleHypothesis(None, Detection([[0]])),
            timestamp=datetime.datetime(2018, 1, 1, 14))
    ])
    tracks = {track}

    deleted_tracks = multi_deleter.delete_tracks(tracks)
    tracks -= deleted_tracks

    if intersect:
        assert len(tracks) == 1
        assert len(deleted_tracks) == 0
    else:
        assert len(tracks) == 0
        assert len(deleted_tracks) == 1

    # Create track that is deleted by both cbd and time deleter
    tracks = {track}
    new_time = datetime.datetime(2018, 1, 1, 14, 25)
    deleted_tracks = multi_deleter.delete_tracks(tracks, timestamp=new_time)
    tracks -= deleted_tracks

    assert (len(tracks) == 0)
    assert (len(deleted_tracks) == 1)
