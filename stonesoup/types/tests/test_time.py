# -*- coding: utf-8 -*-

import datetime

import pytest

from ..time import TimeRange


def test_timerange():
    # Test time range initialisation

    t1 = datetime.datetime(year=2018, month=3, day=1, hour=5, minute=3,
                           second=35, microsecond=500)
    t2 = datetime.datetime(year=2018, month=3, day=1, hour=6, minute=5,
                           second=41, microsecond=500)
    # Test creating without times
    with pytest.raises(TypeError):
        TimeRange()

    # Without start time
    with pytest.raises(TypeError):
        TimeRange(start_timestamp=t1)

    # Without end time
    with pytest.raises(TypeError):
        TimeRange(end_timestamp=t2)

    # Test an error is caught when end is after start
    with pytest.raises(ValueError):
        TimeRange(start_timestamp=t2, end_timestamp=t1)

    test_range = test_range = TimeRange(start_timestamp=t1, end_timestamp=t2)

    assert test_range.start_timestamp == t1
    assert test_range.end_timestamp == t2


def test_duration():
    # Test that duration is calculated properly
    t1 = datetime.datetime(year=2018, month=3, day=1, hour=5, minute=3,
                           second=35, microsecond=500)
    t2 = datetime.datetime(year=2018, month=3, day=1, hour=6, minute=5,
                           second=41, microsecond=500)
    test_range = TimeRange(start_timestamp=t1, end_timestamp=t2)

    assert test_range.duration == datetime.timedelta(seconds=3726)


def test_contains():
    # Test that timestamps are correctly determined to be in the range

    t1 = datetime.datetime(year=2018, month=3, day=1, hour=5, minute=3,
                           second=35, microsecond=500)
    t2 = datetime.datetime(year=2018, month=3, day=1, hour=6, minute=5,
                           second=41, microsecond=500)
    test_range = TimeRange(start_timestamp=t1, end_timestamp=t2)

    # Inside
    assert datetime.datetime(
        year=2018, month=3, day=1, hour=5, minute=10, second=3) in test_range

    # Outside

    assert not datetime.datetime(
        year=2018, month=3, day=1, hour=3, minute=10, second=3) in test_range

    # Lower edge
    assert t1 in test_range

    # Upper edge
    assert t2 in test_range
