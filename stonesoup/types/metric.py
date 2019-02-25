# -*- coding: utf-8 -*-
import datetime

from .base import Type
from .time import TimeRange
from ..base import Property


class Metric(Type):
    """Metric type"""

    title = Property(str, doc='Name of the metric')
    value = Property(any, doc='Value of the metric')
    generator = Property(any, doc='Generator used to create the metric')


class PlottingMetric(Metric):
    """Metric which is to be visualised as plot, value should be a pyplot
     figure"""


class SingleTimeMetric(Metric):
    """Metric for a specific timestamp"""

    timestamp = Property(datetime.datetime, default=None,
                         doc="Timestamp of the state. Default None.")


class TimeRangeMetric(Metric):
    """ Metric for a range of times (e.g. for example an entire run)"""

    time_range = Property(TimeRange, default=None,
                          doc="Time range over which metric assessment will be"
                              "conducted over. Default is None")


class TimeRangePlottingMetric(TimeRangeMetric, PlottingMetric):
    """Plotting metric covering a period of time"""


class SingleTimePlottingMetric(SingleTimeMetric, PlottingMetric):
    """Plotting metric covering a specific timestamp"""
