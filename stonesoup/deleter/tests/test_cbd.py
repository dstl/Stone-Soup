import datetime

import numpy as np

from stonesoup.types import Track, GaussianState
from stonesoup.deleter import CovarianceBasedDeleter


def test_cbd():
    """Test CovarianceBasedDeleter"""

    timestamp = datetime.datetime.now()
    state = GaussianState(
        np.array([[0], [0]]),
        np.array([[100, 0], [0, 1]]), timestamp)
    track = Track()
    track.states.append(state)
    tracks = {track}

    cover_deletion_thresh = 100
    deleter = CovarianceBasedDeleter(cover_deletion_thresh)

    deleted_tracks = deleter.delete_tracks(tracks)
    tracks -= deleted_tracks

    assert(len(tracks) == 0)
    assert(len(deleted_tracks) == 1)
