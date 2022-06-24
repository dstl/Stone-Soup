import datetime

import pytest

from ..time import TimeRange, CompoundTimeRange

@pytest.fixture
def times():
    # Note times are returned chronologically for ease of reading
    before = datetime.datetime(year=2018, month=3, day=1, hour=3, minute=10, second=3)
    t1 = datetime.datetime(year=2018, month=3, day=1, hour=5, minute=3,
                           second=35, microsecond=500)
    inside = datetime.datetime(year=2018, month=3, day=1, hour=5, minute=10, second=3)
    t2 = datetime.datetime(year=2018, month=3, day=1, hour=6, minute=5,
                           second=41, microsecond=500)
    after = datetime.datetime(year=2019, month=3, day=1, hour=6, minute=5,
                              second=41, microsecond=500)
    long_after = datetime.datetime(year=2022, month=6, day=1, hour=6, minute=5,
                                   second=41, microsecond=500)
    return [before, t1, inside, t2, after, long_after]


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

    test_range = TimeRange(start_timestamp=times[1], end_timestamp=times[3])

    test_compound = CompoundTimeRange()

    test_compound2 = CompoundTimeRange([test_range])

    # Tests fuse_components method
    fuse_test = CompoundTimeRange([test_range, TimeRange(times[3], times[4])])

    assert test_range.start_timestamp == times[1]
    assert test_range.end_timestamp == times[3]
    assert len(test_compound.time_ranges) == 0
    assert test_compound2.time_ranges[0] == test_range
    assert fuse_test.time_ranges == [TimeRange(times[1], times[4])]


def test_duration(times):
    # Test that duration is calculated properly

    # TimeRange
    test_range = TimeRange(start_timestamp=times[1], end_timestamp=times[3])

    # CompoundTimeRange

    # times[2] is inside of [1] and [3], so should be equivalent to a TimeRange(times[1], times[4])
    test_range2 = CompoundTimeRange([TimeRange(start_timestamp=times[1], end_timestamp=times[3]),
                                    TimeRange(start_timestamp=times[2], end_timestamp=times[4])])

    assert test_range.duration == datetime.timedelta(seconds=3726)
    assert test_range2.duration == datetime.timedelta(seconds=31539726)


def test_contains(times):
    # Test that timestamps are correctly determined to be in the range

    test_range = TimeRange(start_timestamp=times[1], end_timestamp=times[3])
    test2 = TimeRange(times[1], times[2])
    test3 = TimeRange(times[1], times[4])

    with pytest.raises(TypeError):
        16 in test3

    assert times[2] in test_range
    assert not times[4] in test_range
    assert not times[0] in test_range
    assert times[1] in test_range
    assert times[3] in test_range

    assert test2 in test_range
    assert test_range not in test2
    assert test2 in test2
    assert test3 not in test_range

    # CompoundTimeRange

    compound_test = CompoundTimeRange([test_range])
    # Should be in neither
    test_range2 = TimeRange(times[4], times[5])
    # Should be in compound_range2 but not 1
    test_range3 = TimeRange(times[2], times[4])
    compound_test2 = CompoundTimeRange([test_range, TimeRange(times[3], times[4])])

    assert compound_test in compound_test2
    assert times[2] in compound_test
    assert times[2] in compound_test2
    assert test_range2 not in compound_test
    assert test_range2 not in compound_test2
    assert test_range3 not in compound_test
    assert test_range3 in compound_test2


def test_equality(times):
    test1 = TimeRange(times[1], times[2])
    test2 = TimeRange(times[1], times[2])
    test3 = TimeRange(times[1], times[3])

    with pytest.raises(TypeError):
        test1 == "stonesoup"

    assert test1 == test2
    assert test2 == test1
    assert test3 != test1 and test1 != test3

    ctest1 = CompoundTimeRange([test1, test3])
    ctest2 = CompoundTimeRange([TimeRange(times[1], times[3])])

    with pytest.raises(TypeError):
        ctest2 == "Stonesoup is the best!"

    assert ctest1 == ctest2
    assert ctest2 == ctest1
    ctest2.add(TimeRange(times[3], times[4]))
    assert ctest1 != ctest2
    assert ctest2 != ctest1

    assert CompoundTimeRange() == CompoundTimeRange()
    assert ctest1 != CompoundTimeRange()
    assert CompoundTimeRange() != ctest1


def test_minus(times):
    # Test the minus function
    test1 = TimeRange(times[1], times[3])
    test2 = TimeRange(times[1], times[2])
    test3 = TimeRange(times[2], times[3])
    test4 = TimeRange(times[4], times[5])

    with pytest.raises(TypeError):
        test1.minus(15)

    assert test1.minus(test2) == test3
    assert test1.minus(None) == test1
    assert test2.minus(test1) is None

    ctest1 = CompoundTimeRange([test2, test4])
    ctest2 = CompoundTimeRange([test1, test2])
    ctest3 = CompoundTimeRange([test4])

    with pytest.raises(TypeError):
        ctest1.minus(15)

    assert ctest1.minus(ctest2) == ctest3
    assert ctest1.minus(ctest1) == CompoundTimeRange()
    assert ctest3.minus(ctest1) == CompoundTimeRange()

    assert test1.minus(ctest1) == TimeRange(times[2], times[3])
    assert test4.minus(ctest2) == test4
    assert ctest1.minus(test2) == ctest3


def test_overlap(times):
    test1 = TimeRange(times[1], times[3])
    test2 = TimeRange(times[1], times[2])
    test3 = TimeRange(times[4], times[5])

    ctest1 = CompoundTimeRange([test2, test3])
    ctest2 = CompoundTimeRange([test1, test2])

    with pytest.raises(TypeError):
        test2.overlap(ctest1)

    assert test1.overlap(test1) == test1
    assert test1.overlap(None) is None
    assert test1.overlap(test2) == test2
    assert test2.overlap(test1) == test2
    assert ctest1.overlap(None) is None
    assert ctest1.overlap(test2) == CompoundTimeRange([test2])
    assert ctest1.overlap(ctest2) == CompoundTimeRange([test2])
    assert ctest1.overlap(ctest2) == ctest2.overlap(ctest1)


def test_key_times(times):
    test1 = CompoundTimeRange([TimeRange(times[0], times[1]),
                               TimeRange(times[3], times[4])])
    test2 = CompoundTimeRange([TimeRange(times[3], times[4]),
                               TimeRange(times[0], times[1])])
    test3 = CompoundTimeRange()
    test4 = CompoundTimeRange([TimeRange(times[0], times[4])])

    assert test1.key_times == [times[0], times[1], times[3], times[4]]
    assert test2.key_times == [times[0], times[1], times[3], times[4]]
    assert test3.key_times == []
    assert test4.key_times == [times[0], times[4]]


def test_remove_overlap(times):
    test1_ro = CompoundTimeRange([TimeRange(times[0], times[1]),
                                  TimeRange(times[3], times[4])])
    test1_ro._remove_overlap()
    test2_ro = CompoundTimeRange([TimeRange(times[3], times[4]),
                                  TimeRange(times[0], times[4])])
    test2_ro._remove_overlap()
    test3_ro = CompoundTimeRange()
    test3_ro._remove_overlap()

    test1 = CompoundTimeRange([TimeRange(times[0], times[1]),
                               TimeRange(times[3], times[4])])
    test3 = CompoundTimeRange()
    test4 = CompoundTimeRange([TimeRange(times[0], times[4])])

    assert test1_ro == test1
    assert test2_ro == test4
    assert test3_ro == test3


def test_fuse_components(times):
    # Note this is called inside the __init__ method, but is tested here explicitly
    test1 = CompoundTimeRange([TimeRange(times[1], times[2])])._fuse_components()
    test2 = CompoundTimeRange([TimeRange(times[1], times[2]),
                               TimeRange(times[2], times[4])])._fuse_components()
    assert test1.time_ranges == {TimeRange(times[1], times[2])}
    assert test2.time_ranges == {TimeRange(times[1], times[4])}

def test_add(times):
    test1 = CompoundTimeRange([TimeRange(times[1], times[2])])
    test2 = CompoundTimeRange([TimeRange(times[0], times[1])])
    test3 = CompoundTimeRange([TimeRange(times[0], times[2])])
    test4 = CompoundTimeRange([TimeRange(times[0], times[2]),
                               TimeRange(times[4], times[5])])
    with pytest.raises(TypeError):
        test1.add(True)
    assert test1 != test2
    test1_copy = test1
    test1.add(None)
    assert test1 == test1_copy
    test1.add(test2)
    assert test1 == test3
    test3.add(TimeRange(times[4], times[5]))
    assert test3 == test4

def test_remove(times):
    test1 = CompoundTimeRange([TimeRange(times[0], times[2]),
                               TimeRange(times[4], times[5])])
    with pytest.raises(TypeError):
        test1.remove(0.4)
    with pytest.raises(ValueError):
        test1.remove(TimeRange(times[0], times[1]))

    test1.remove(TimeRange(times[0], times[2]))
    assert test1 == CompoundTimeRange([TimeRange(times[4], times[5])])
