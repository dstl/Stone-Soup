import datetime

import numpy as np

from ..error import CovarianceBasedDeleter
from ...types.state import GaussianState
from ...types.track import Track


def test_cbd():
    """Test CovarianceBasedDeleter"""

    timestamp = datetime.datetime.now()
    state = GaussianState(
        np.array([[0], [0]]),
        np.array([[100, 0], [0, 1]]), timestamp)
    track1 = Track(state)

    state = GaussianState(
        np.array([[0], [0]]),
        np.array([[1, 0], [0, 1]]), timestamp)
    track2 = Track(state)

    tracks = {track1, track2}

    cover_deletion_thresh = 100
    deleter = CovarianceBasedDeleter(covar_trace_thresh=cover_deletion_thresh)

    deleted_tracks = deleter.delete_tracks(tracks)
    tracks -= deleted_tracks

    assert len(tracks) == 1
    assert len(deleted_tracks) == 1

    deleter = CovarianceBasedDeleter(cover_deletion_thresh, mapping=[1])

    tracks = {track1, track2}

    deleted_tracks = deleter.delete_tracks(tracks)
    tracks -= deleted_tracks

    assert len(tracks) == 2
    assert len(deleted_tracks) == 0
