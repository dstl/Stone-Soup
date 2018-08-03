# -*- coding: utf-8 -*-
import datetime

from .base import Type
from .base import Property

class Metric(Type):
    """Metric type"""

    title = Property(str, doc = 'Name of the metric')
    value = Property(any, doc = 'Value of the metric')
    generator = Property(None, doc = 'Generator used to create the metric')

class SingleTimeMetric(Metric):
    """ Metric for a specific timestamp"""

    timestamp = Property(datetime.datetime, default=None,
                         doc="Timestamp of the state. Default None.")

class TimePeriodMetric(Metric):
    """ Metric for a range of times (for example an entire run)"""

    start_timestamp = Property(datetime.datetime, doc = 'Start of the time period that the metric covers')
    end_timestamp = Property(datetime.datetime, doc = 'End of the time period that the metric covers')
