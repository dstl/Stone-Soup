import datetime

import pytest

from ..time import TimeRange, CompoundTimeRange

@pytest.fixture
def times():
    before = datetime.datetime(year=2018, month=3, day=1, hour=3, minute=10, second=3)
    t1 = datetime.datetime(year=2018, month=3, day=1, hour=5, minute=3,
                           second=35, microsecond=500)
    inside = datetime.datetime(year=2018, month=3, day=1, hour=5, minute=10, second=3)
    t2 = datetime.datetime(year=2018, month=3, day=1, hour=6, minute=5,
                           second=41, microsecond=500)
    after = datetime.datetime(year=2019, month=3, day=1, hour=6, minute=5,
                              second=41, microsecond=500)
    return [before, t1, inside, t2, after]


def test_timerange(times):
    # Test creating without times
    with pytest.raises(TypeError):
        TimeRange()

    # Without start time
    with pytest.raises(TypeError):
        TimeRange(start_timestamp=times[1])

    # Without end time
    with pytest.raises(TypeError):
        TimeRange(end_timestamp=times[3])

    # Test an error is caught when end is after start
    with pytest.raises(ValueError):
        TimeRange(start_timestamp=times[3], end_timestamp=times[1])

    # Test with wrong types for time_ranges
    with pytest.raises(TypeError):
        CompoundTimeRange(42)
    with pytest.raises(TypeError):
        CompoundTimeRange([times[1], times[3]])

    test_range = test_range = TimeRange(start_timestamp=times[1], end_timestamp=times[3])

    test_compound = CompoundTimeRange()

    test_compound2 = CompoundTimeRange([test_range])

    assert test_range.start_timestamp == times[1]
    assert test_range.end_timestamp == times[3]
    assert len(test_compound.time_ranges) == 0
    assert test_compound2.time_ranges[0] == test_range


def test_duration(times):
    # Test that duration is calculated properly

    # TimeRange
    test_range = TimeRange(start_timestamp=times[1], end_timestamp=times[3])

    # CompoundTimeRange

    # times[2] is inside of [1] and [3], so should be equivalent to a TimeRange(times[1], times[4])
    test_range2 = CompoundTimeRange(TimeRange(start_timestamp=times[1], end_timestamp=times[3]),
                                    TimeRange(start_timestamp=times[2], end_timestamp=times[4]))

    assert test_range.duration == datetime.timedelta(seconds=3726)
    assert test_range.duration == datetime.timedelta(seconds=31539726)


def test_timerange_contains():
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

def test_timerange_minus():
    # Test the minus function
    test1 = TimeRange()
