# -*- coding: utf-8 -*-
import datetime

from ..time import UpdateTimeStepsDeleter, UpdateTimeDeleter
from ...types.detection import Detection
from ...types.hypothesis import SingleHypothesis
from ...types.prediction import StatePrediction
from ...types.track import Track
from ...types.update import StateUpdate


def test_update_time_steps_deleter():
    deleter = UpdateTimeStepsDeleter(3)

    track = Track([
        StateUpdate(
            [[0]],
            SingleHypothesis(None, Detection([[0]])),
            timestamp=datetime.datetime(2018, 1, 1, 14)),
        StatePrediction(
            [[0]], timestamp=datetime.datetime(2018, 1, 1, 14, 10)),
        StatePrediction(
            [[0]], timestamp=datetime.datetime(2018, 1, 1, 14, 20)),
    ])

    # Last update within time steps
    tracks2delete = deleter.delete_tracks({track})
    assert not tracks2delete

    # Add new state (taking outside time)
    track.append(StatePrediction(
        [[0]], timestamp=datetime.datetime(2018, 1, 1, 14, 30)))

    # Last update outside time step
    tracks2delete = deleter.delete_tracks({track})
    assert tracks2delete

    # Add new update without measurement
    track.append(StateUpdate(
        [[0]],
        SingleHypothesis(None, None),
        timestamp=datetime.datetime(2018, 1, 1, 14, 30)))
    tracks2delete = deleter.delete_tracks({track})
    assert tracks2delete


def test_update_time_deleter():
    deleter = UpdateTimeDeleter(datetime.timedelta(minutes=20))

    track = Track([
        StateUpdate(
            [[0]],
            SingleHypothesis(None, Detection([[0]])),
            timestamp=datetime.datetime(2018, 1, 1, 14)),
        StatePrediction(
            [[0]], timestamp=datetime.datetime(2018, 1, 1, 14, 10)),
        StatePrediction(
            [[0]], timestamp=datetime.datetime(2018, 1, 1, 14, 20)),
    ])

    # Last update within time
    tracks2delete = deleter.delete_tracks({track})
    assert not tracks2delete

    # Last update within custom time
    tracks2delete = deleter.delete_tracks(
        {track}, timestamp=datetime.datetime(2018, 1, 1, 14, 15))
    assert not tracks2delete

    # Last update outside custom time
    tracks2delete = deleter.delete_tracks(
        {track}, timestamp=datetime.datetime(2018, 1, 1, 14, 25))
    assert tracks2delete

    # Add new state (taking outside time)
    track.append(StatePrediction(
        [[0]], timestamp=datetime.datetime(2018, 1, 1, 14, 30)))

    # Last update outside time
    tracks2delete = deleter.delete_tracks({track})
    assert tracks2delete

    # Add new update without measurement
    track.append(StateUpdate(
        [[0]],
        SingleHypothesis(None, None),
        timestamp=datetime.datetime(2018, 1, 1, 14, 30)))
    tracks2delete = deleter.delete_tracks({track})
    assert tracks2delete

    # Last update within custom time
    tracks2delete = deleter.delete_tracks(
        {track}, timestamp=datetime.datetime(2018, 1, 1, 14, 15))
    assert not tracks2delete
