import datetime

import numpy as np

from ..manager import SimpleManager
from ..basicmetrics import BasicMetrics
from ...types.groundtruth import GroundTruthPath
from ...types.metric import TimeRangeMetric
from ...types.state import State
from ...types.time import TimeRange
from ...types.track import Track


def test_basicmetrics():
    generator = BasicMetrics()
    manager = SimpleManager([generator])

    start_time = datetime.datetime.now()
    tracks = set(Track(
        states=[State(np.array([[i], [j]]),
                      timestamp=start_time + datetime.timedelta(seconds=i))
                for i in range(5)]) for j in range(4))

    truths = set(GroundTruthPath(
        states=[State(np.array([[i], [j]]),
                      timestamp=start_time + datetime.timedelta(seconds=i))
                for i in range(5)]) for j in range(3))

    manager.add_data([tracks, truths])

    metrics = manager.generate_metrics()

    correct_metrics = {TimeRangeMetric(title='Number of targets',
                                       value=3,
                                       time_range=TimeRange(
                                           start_timestamp=start_time,
                                           end_timestamp=start_time +
                                           datetime.timedelta(seconds=4)),
                                       generator=generator),
                       TimeRangeMetric(title='Number of tracks',
                                       value=4,
                                       time_range=TimeRange(
                                           start_timestamp=start_time,
                                           end_timestamp=start_time +
                                           datetime.timedelta(seconds=4)),
                                       generator=generator),
                       TimeRangeMetric(title='Track-to-target ratio',
                                       value=4 / 3,
                                       time_range=TimeRange(
                                           start_timestamp=start_time,
                                           end_timestamp=start_time +
                                           datetime.timedelta(seconds=4)),
                                       generator=generator)}

    for metric_name in ["Number of targets",
                        "Number of tracks", "Track-to-target ratio"]:
        calc_metric = [i for i in correct_metrics if i.title == metric_name][0]
        meas_metric = [i for i in metrics if i.title == metric_name][0]
        assert calc_metric.value == meas_metric.value
        assert calc_metric.time_range.start_timestamp == \
            meas_metric.time_range.start_timestamp
        assert calc_metric.time_range.end_timestamp == \
            meas_metric.time_range.end_timestamp
        assert calc_metric.generator == meas_metric.generator
