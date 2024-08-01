import datetime

import pytest

from ..time import TimeBufferedFeeder, TimeSyncFeeder


def test_time_buffered_feeder_detections(detector):
    feeder = TimeBufferedFeeder(detector)

    prev_time = datetime.datetime(2019, 4, 1, 13, 59, 59)
    for steps, (time, detections) in enumerate(feeder, 1):
        assert time > prev_time
        prev_time = time

    assert steps == 7


def test_time_buffered_feeder_groundtruth(groundtruth):
    feeder = TimeBufferedFeeder(groundtruth)

    prev_time = datetime.datetime(2019, 12, 31, 23, 59, 59)
    for steps, (time, truths) in enumerate(feeder, 1):
        assert time > prev_time

    assert steps == 3


def test_time_buffered_feeder_buffer_size_detections(detector):
    feeder = TimeBufferedFeeder(detector, buffer_size=1)
    with pytest.warns(UserWarning):
        # Only 6 steps as one skipped due to being out of order
        assert sum(1 for _ in feeder) == 6

    feeder = TimeBufferedFeeder(detector, buffer_size=2)
    # All 7 steps as buffer large enough
    assert sum(1 for _ in feeder) == 7


def test_time_buffered_feeder_buffer_size_groundtruth(groundtruth):
    feeder = TimeBufferedFeeder(groundtruth, buffer_size=1)
    with pytest.warns(UserWarning):
        # Only 2 steps as one skipped due to being out of order
        assert sum(1 for _ in feeder) == 2

    feeder = TimeBufferedFeeder(groundtruth, buffer_size=2)
    # All 3 steps as buffer large enough
    assert sum(1 for _ in feeder) == 3


def test_time_sync_feeder_detections(detector):
    feeder = TimeSyncFeeder(detector, datetime.timedelta(seconds=2))

    prev_time = datetime.datetime(2019, 4, 1, 14) - feeder.time_window
    for steps, (time, detections) in enumerate(feeder, 1):
        assert time == prev_time + feeder.time_window
        assert all(detection.timestamp >= prev_time
                   for detection in detections)
        prev_time = time

    assert steps == 4


def test_time_sync_feeder_groundtruth(groundtruth):
    feeder = TimeSyncFeeder(groundtruth, datetime.timedelta(seconds=2))

    prev_time = datetime.datetime(2020, 1, 1, 0) - feeder.time_window
    for steps, (time, detections) in enumerate(feeder, 1):
        assert time == prev_time + feeder.time_window
        assert all(detection.timestamp >= prev_time
                   for detection in detections)
        prev_time = time

    assert steps == 2
