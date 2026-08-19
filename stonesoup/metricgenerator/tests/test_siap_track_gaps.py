import datetime

import pytest

from ...measures import Euclidean
from ...types.association import AssociationSet, TimeRangeAssociation
from ...types.groundtruth import GroundTruthPath, GroundTruthState
from ...types.state import State
from ...types.time import TimeRange
from ...types.track import Track
from ..manager import MultiManager
from ..tracktotruthmetrics import SIAPMetrics


def _manager_with_association(truth, track, start_time, end_time):
    manager = MultiManager()
    manager.add_data({
        'groundtruth_paths': [truth],
        'tracks': [track],
    })
    manager.association_set = AssociationSet({
        TimeRangeAssociation(
            {truth, track},
            time_range=TimeRange(start_time, end_time),
        )
    })
    return manager


def _metric(manager):
    metric = SIAPMetrics(
        position_measure=Euclidean((0, 2)),
        velocity_measure=Euclidean((1, 3)),
    )
    manager.generators = [metric]
    return metric


def test_siap_interpolates_association_timestamp_without_track_state():
    t0 = datetime.datetime(2024, 1, 1, 0, 0, 0)
    t1 = t0 + datetime.timedelta(seconds=1)
    t2 = t0 + datetime.timedelta(seconds=2)

    truth = GroundTruthPath([
        GroundTruthState([0, 0, 0, 0], timestamp=t0),
        GroundTruthState([1, 0, 1, 0], timestamp=t1),
        GroundTruthState([2, 0, 2, 0], timestamp=t2),
    ])
    track = Track([
        State([0.1, 0, 0.1, 0], timestamp=t0),
        State([2.1, 0, 2.1, 0], timestamp=t2),
    ])

    manager = _manager_with_association(truth, track, t0, t2)
    metric = _metric(manager)
    expected_accuracy = (2 * 0.1 ** 2) ** 0.5

    assert metric.num_tracks_at_time([track], t1) == 1
    assert metric.num_associated_tracks_at_time(manager, [track], t1) == 1
    assert metric.accuracy_at_time(
        manager, t1, metric.position_measure) == pytest.approx(expected_accuracy)

    metrics = metric.compute_metric(manager)
    position_at_times = next(
        item for item in metrics if item.title == 'SIAP Position Accuracy at times'
    )
    gap_metric = next(item for item in position_at_times.value if item.timestamp == t1)

    assert gap_metric.value == pytest.approx(expected_accuracy)


def test_siap_interpolates_association_timestamp_without_truth_state():
    t0 = datetime.datetime(2024, 1, 1, 0, 0, 0)
    t1 = t0 + datetime.timedelta(seconds=1)
    t2 = t0 + datetime.timedelta(seconds=2)

    truth = GroundTruthPath([
        GroundTruthState([0, 0, 0, 0], timestamp=t0),
        GroundTruthState([2, 0, 2, 0], timestamp=t2),
    ])
    track = Track([
        State([0.1, 0, 0.1, 0], timestamp=t0),
        State([1.1, 0, 1.1, 0], timestamp=t1),
        State([2.1, 0, 2.1, 0], timestamp=t2),
    ])

    manager = _manager_with_association(truth, track, t0, t2)
    metric = _metric(manager)
    expected_accuracy = (2 * 0.1 ** 2) ** 0.5

    assert metric.num_truths_at_time([truth], t1) == 1
    assert metric.num_associated_truths_at_time(manager, [truth], t1) == 1
    assert metric.accuracy_at_time(
        manager, t1, metric.position_measure) == pytest.approx(expected_accuracy)
