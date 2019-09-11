from matplotlib.table import Table

from ...types.metric import TimeRangeMetric
from ...types.time import TimeRange
from ..metrictables import SIAPTableGenerator
from ..tracktotruthmetrics import SIAPMetrics


def test_siaptable():
    metric_generator = SIAPMetrics()
    metrics = set()
    # Add metrics to ensure the generator can handle a full range of values
    metrics.add(TimeRangeMetric(title="SIAP C",
                                value=1,
                                time_range=TimeRange(0, 10),
                                generator=metric_generator))
    metrics.add(TimeRangeMetric(title="SIAP A",
                                value=0.65,
                                time_range=TimeRange(0, 10),
                                generator=metric_generator))
    metrics.add(TimeRangeMetric(title="SIAP S",
                                value=0.35,
                                time_range=TimeRange(0, 10),
                                generator=metric_generator))
    metrics.add(TimeRangeMetric(title="SIAP LT",
                                value=12,
                                time_range=TimeRange(0, 10),
                                generator=metric_generator))
    metrics.add(TimeRangeMetric(title="SIAP LS",
                                value=0,
                                time_range=TimeRange(0, 10),
                                generator=metric_generator))
    # Generate table
    table_generator = SIAPTableGenerator(metrics)
    table = table_generator.generate_table()
    assert isinstance(table, Table)
