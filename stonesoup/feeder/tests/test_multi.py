import time
from datetime import datetime, timedelta

import pytest

from ..multi import (
    MultiDataFeeder, FIFOMultiDataFeeder, LIFOMultiDataFeeder,
    PriorityMultiDataFeeder, MaxSizePriorityMultiDataFeeder)


@pytest.mark.parametrize('n', [1, 2, 3])
def test_multi_detections(n, reader):
    single_time_list = list()
    multi_time_list = list()

    multi_detector = MultiDataFeeder([reader]*n)
    for single_iterations, (timestamp, _) in enumerate(reader, 1):
        single_time_list.append(timestamp)
    for multi_iterations, (timestamp, _) in enumerate(multi_detector, 1):
        multi_time_list.append(timestamp)
    assert multi_iterations == single_iterations * n
    # Compare every other element but skip last ones due to out of sequence
    # measurements coming from detector.
    assert multi_time_list[:-n:n] == single_time_list[:-1]
    assert multi_time_list[-1] == single_time_list[-1]


@pytest.fixture(scope="function")
def readers():
    start_time = datetime(2025, 3, 26)

    def reader1():
        timestamp = start_time
        yield timestamp, {1}
        time.sleep(0.2)
        timestamp += timedelta(hours=2)
        yield timestamp, {1}
        time.sleep(0.2)
        timestamp += timedelta(hours=2)
        yield timestamp, {1}
        time.sleep(0.4)
        timestamp += timedelta(hours=1)
        yield timestamp, {1}

    def reader2():
        timestamp = start_time
        time.sleep(0.1)
        timestamp += timedelta(hours=1)
        yield timestamp, {2}
        time.sleep(0.2)
        timestamp += timedelta(hours=2)
        yield timestamp, {2}
        time.sleep(0.2)
        timestamp += timedelta(hours=3)
        yield timestamp, {2}
        time.sleep(0.1)
        timestamp += timedelta(hours=1)
        yield timestamp, {2}

    return [reader1(), reader2()]


def test_fifo_feeder(readers):
    feeder = FIFOMultiDataFeeder(readers)
    assert list((t, d) for t, d in feeder if not time.sleep(0.21)) == [
        (datetime(2025, 3, 26, 0), {1}),
        (datetime(2025, 3, 26, 1), {2}),
        (datetime(2025, 3, 26, 2), {1}),
        (datetime(2025, 3, 26, 3), {2}),
        (datetime(2025, 3, 26, 4), {1}),
        (datetime(2025, 3, 26, 6), {2}),
        (datetime(2025, 3, 26, 7), {2}),
        (datetime(2025, 3, 26, 5), {1}),
    ]


def test_lifo_feeder(readers):
    feeder = LIFOMultiDataFeeder(readers)
    assert list((t, d) for t, d in feeder if not time.sleep(0.21)) == [
        (datetime(2025, 3, 26, 0), {1}),
        (datetime(2025, 3, 26, 2), {1}),
        (datetime(2025, 3, 26, 4), {1}),
        (datetime(2025, 3, 26, 7), {2}),
        (datetime(2025, 3, 26, 5), {1}),
        (datetime(2025, 3, 26, 6), {2}),
        (datetime(2025, 3, 26, 3), {2}),
        (datetime(2025, 3, 26, 1), {2}),
    ]


def test_priority_feeder(readers):
    feeder = PriorityMultiDataFeeder(readers)
    assert list((t, d) for t, d in feeder if not time.sleep(0.21)) == [
        (datetime(2025, 3, 26, 0), {1}),
        (datetime(2025, 3, 26, 1), {2}),
        (datetime(2025, 3, 26, 2), {1}),
        (datetime(2025, 3, 26, 3), {2}),
        (datetime(2025, 3, 26, 4), {1}),
        (datetime(2025, 3, 26, 5), {1}),
        (datetime(2025, 3, 26, 6), {2}),
        (datetime(2025, 3, 26, 7), {2}),
    ]


def test_max_priority_feeder(readers):
    feeder = MaxSizePriorityMultiDataFeeder(readers, 3)
    assert list((t, d) for t, d in feeder if not time.sleep(0.21)) == [
        (datetime(2025, 3, 26, 0), {1}),
        (datetime(2025, 3, 26, 1), {2}),
        (datetime(2025, 3, 26, 2), {1}),
        # (datetime(2025, 3, 26, 3), {2}), # Dropped before consumed
        (datetime(2025, 3, 26, 4), {1}),
        (datetime(2025, 3, 26, 5), {1}),
        (datetime(2025, 3, 26, 6), {2}),
        (datetime(2025, 3, 26, 7), {2}),
    ]


def test_no_wait(readers):
    # Testing code handles empty queues
    feeder = PriorityMultiDataFeeder(readers)
    # Same as FIFO as not allowing time to prioritise
    assert list((t, d) for t, d in feeder) == [
        (datetime(2025, 3, 26, 0), {1}),
        (datetime(2025, 3, 26, 1), {2}),
        (datetime(2025, 3, 26, 2), {1}),
        (datetime(2025, 3, 26, 3), {2}),
        (datetime(2025, 3, 26, 4), {1}),
        (datetime(2025, 3, 26, 6), {2}),
        (datetime(2025, 3, 26, 7), {2}),
        (datetime(2025, 3, 26, 5), {1}),
    ]
