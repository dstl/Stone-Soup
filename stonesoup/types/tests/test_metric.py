import datetime

import pytest

from ...metricgenerator import MetricGenerator
from ..metric import (
    Metric,
    PlottingMetric,
    SingleTimeMetric,
    SingleTimePlottingMetric,
    TimeRangeMetric,
    TimeRangePlottingMetric,
)
from ..time import TimeRange


class TempGenerator(MetricGenerator):
    generator_name = "temp-generator"

    def compute_metric(self, manager):
        return 5


def test_metric():
    with pytest.raises(TypeError):
        Metric()

    title = "Test metric"
    value = 5

    generator = TempGenerator()

    temp_metric = Metric(title=title,
                         value=value,
                         generator=generator)

    assert temp_metric.title == title
    assert temp_metric.value == value
    assert temp_metric.generator == generator


def test_plottingmetric():
    with pytest.raises(TypeError):
        PlottingMetric()

    title = "Test metric"
    value = 5

    generator = TempGenerator()
    temp_metric = PlottingMetric(title=title,
                                 value=value,
                                 generator=generator)

    assert temp_metric.title == title
    assert temp_metric.value == value
    assert temp_metric.generator == generator


def test_singletimemetric():
    with pytest.raises(TypeError):
        SingleTimeMetric()

    title = "Test metric"
    value = 5
    timestamp = datetime.datetime.now()

    generator = TempGenerator()
    temp_metric = SingleTimeMetric(title=title,
                                   value=value,
                                   timestamp=timestamp,
                                   generator=generator)

    assert temp_metric.title == title
    assert temp_metric.value == value
    assert temp_metric.timestamp == timestamp
    assert temp_metric.generator == generator


def test_timerangemetric():
    with pytest.raises(TypeError):
        TimeRangeMetric()

    title = "Test metric"
    value = 5
    timestamp1 = datetime.datetime.now()
    timestamp2 = timestamp1 + datetime.timedelta(seconds=10)
    time_range = TimeRange(start=timestamp1,
                           end=timestamp2)

    generator = TempGenerator()
    temp_metric = TimeRangeMetric(title=title,
                                  value=value,
                                  time_range=time_range,
                                  generator=generator)

    assert temp_metric.title == title
    assert temp_metric.value == value
    assert temp_metric.time_range == time_range
    assert temp_metric.generator == generator


def test_timerangeplottingmetric():
    with pytest.raises(TypeError):
        TimeRangePlottingMetric()

    title = "Test metric"
    value = 5
    timestamp1 = datetime.datetime.now()
    timestamp2 = timestamp1 + datetime.timedelta(seconds=10)
    time_range = TimeRange(start=timestamp1,
                           end=timestamp2)

    generator = TempGenerator()
    temp_metric = TimeRangePlottingMetric(title=title,
                                          value=value,
                                          time_range=time_range,
                                          generator=generator)

    assert temp_metric.title == title
    assert temp_metric.value == value
    assert temp_metric.time_range == time_range
    assert temp_metric.generator == generator


def test_single_timeplottingmetric():
    with pytest.raises(TypeError):
        SingleTimePlottingMetric()

    title = "Test metric"
    value = 5
    timestamp = datetime.datetime.now()

    generator = TempGenerator()
    temp_metric = SingleTimePlottingMetric(title=title,
                                           value=value,
                                           timestamp=timestamp,
                                           generator=generator)

    assert temp_metric.title == title
    assert temp_metric.value == value
    assert temp_metric.timestamp == timestamp
    assert temp_metric.generator == generator
