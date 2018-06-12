import datetime

import numpy as np

from stonesoup.types import Track, GaussianState
from stonesoup.deletor import CovarianceBasedDeletor


def test_cbd():
    """Test CovarianceBasedDeletor"""

    timestamp = datetime.datetime.now()
    state = GaussianState(
        np.array([[0], [0]]),
        np.array([[100, 0], [0, 1]]), timestamp)
    track = Track()
    track.states.append(state)
    tracks = {track}

    cover_deletion_thresh = 100
    deletor = CovarianceBasedDeletor(cover_deletion_thresh)

    deleted_tracks = deletor.delete_tracks(tracks)
    tracks -= deleted_tracks

    assert(len(tracks) == 0)
    assert(len(deleted_tracks) == 1)
