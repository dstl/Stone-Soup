# -*- coding: utf-8 -*-
import datetime

import pytest

from ...deleter import Deleter
from ...types.detection import Detection
from ...types.hypothesis import SingleHypothesis
from ...types.prediction import StatePrediction
from ...types.track import Track
from ...types.update import StateUpdate


class BasicDeleter(Deleter):
    """Deletes tracks if they are longer than 3 states"""

    def check_for_deletion(self, track):
        return len(track) > 3


@pytest.mark.parametrize("delete_last_pred", [True, False])
def test_delete_tracks(delete_last_pred):
    start = datetime.datetime.now()
    times = [start + datetime.timedelta(seconds=5*i) for i in range(4)]

    track1 = Track([
        StateUpdate([[0]], SingleHypothesis(None, Detection([[0]])), timestamp=times[0]),
        StatePrediction([[0]], timestamp=times[1]),
        StatePrediction([[0]], timestamp=times[2]),
        StatePrediction([[0]], timestamp=times[3])
    ])
    track2 = Track([
        StateUpdate([[0]], SingleHypothesis(None, Detection([[0]])), timestamp=times[0]),
        StatePrediction([[0]], timestamp=times[1]),
        StatePrediction([[0]], timestamp=times[2]),
        StateUpdate([[0]], SingleHypothesis(None, Detection([[0]])), timestamp=times[3])
    ])
    track3 = Track([
        StateUpdate([[0]], SingleHypothesis(None, Detection([[0]])), timestamp=times[0]),
        StatePrediction([[0]], timestamp=times[3])
    ])
    track4 = Track([
        StateUpdate([[0]], SingleHypothesis(None, Detection([[0]])), timestamp=times[0]),
        StateUpdate([[0]], SingleHypothesis(None, Detection([[0]])), timestamp=times[3])
    ])

    tracks = {track1, track2, track3, track4}

    deleter = BasicDeleter(delete_last_pred)

    tracks_to_delete = deleter.delete_tracks(tracks)

    tracks -= tracks_to_delete

    assert tracks == {track3, track4}
    assert tracks_to_delete == {track1, track2}

    if delete_last_pred:
        assert len(track1) == 3  # last state is prediction, so delete
    else:
        assert len(track1) == 4
    assert len(track2) == 4  # last state is update, so do not delete
