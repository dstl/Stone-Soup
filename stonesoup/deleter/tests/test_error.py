# -*- coding: utf-8 -*-
import datetime

import numpy as np

from ...types.state import GaussianState
from ...types.track import Track
from ..error import CovarianceBasedDeleter


def test_cbd():
    """Test CovarianceBasedDeleter"""

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

    cover_deletion_thresh = 100
    deleter = CovarianceBasedDeleter(covar_trace_thresh=cover_deletion_thresh)

    deleted_tracks = deleter.delete_tracks(tracks)
    tracks -= deleted_tracks

    assert(len(tracks) == 1)
    assert(len(deleted_tracks) == 1)
