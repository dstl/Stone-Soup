import datetime

from ...measures import Euclidean
from ...types.association import AssociationSet, TimeRangeAssociation
from ...types.groundtruth import GroundTruthPath, GroundTruthState
from ...types.state import State
from ...types.time import TimeRange
from ...types.track import Track
from ..manager import MultiManager
from ..tracktotruthmetrics import SIAPMetrics


def test_siap_ignores_association_timestamp_without_track_state():
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

    manager = MultiManager()
    manager.add_data({
        'groundtruth_paths': [truth],
        'tracks': [track],
    })
    manager.association_set = AssociationSet({
        TimeRangeAssociation(
            {truth, track},
            time_range=TimeRange(t0, t2),
        )
    })

    metric = SIAPMetrics(
        position_measure=Euclidean((0, 2)),
        velocity_measure=Euclidean((1, 3)),
    )
    manager.generators = [metric]

    assert metric.num_tracks_at_time([track], t1) == 0
    assert metric.num_associated_tracks_at_time(manager, [track], t1) == 0
    assert metric.accuracy_at_time(manager, t1, metric.position_measure) == 0

    metrics = metric.compute_metric(manager)
    position_at_times = next(
        item for item in metrics if item.title == 'SIAP Position Accuracy at times'
    )
    gap_metric = next(item for item in position_at_times.value if item.timestamp == t1)

    assert gap_metric.value == 0
