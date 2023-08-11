from matplotlib.table import Table

from ..metrictables import SIAPTableGenerator
from ..tracktotruthmetrics import SIAPMetrics
from ...measures import Euclidean
from ...types.metric import TimeRangeMetric
from ...types.time import TimeRange


def test_siaptable():
    metric_generator = SIAPMetrics(position_measure=Euclidean((0, 2)),
                                   velocity_measure=Euclidean((1, 3)))
    metrics = set()
    # Add metrics to ensure the generator can handle a full range of values
    metrics.add(TimeRangeMetric(title="SIAP Completeness",
                                value=1,
                                time_range=TimeRange(0, 10),
                                generator=metric_generator))
    metrics.add(TimeRangeMetric(title="SIAP Ambiguity",
                                value=0.65,
                                time_range=TimeRange(0, 10),
                                generator=metric_generator))
    metrics.add(TimeRangeMetric(title="SIAP Spuriousness",
                                value=0.35,
                                time_range=TimeRange(0, 10),
                                generator=metric_generator))
    metrics.add(TimeRangeMetric(title="SIAP Position Accuracy",
                                value=12,
                                time_range=TimeRange(0, 10),
                                generator=metric_generator))
    metrics.add(TimeRangeMetric(title="SIAP Velocity Accuracy",
                                value=0,
                                time_range=TimeRange(0, 10),
                                generator=metric_generator))
    # Generate table
    table_generator = SIAPTableGenerator(metrics)
    table = table_generator.compute_metric()
    assert isinstance(table, Table)
