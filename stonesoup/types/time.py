# -*- coding: utf-8 -*-
import datetime

from ..base import Property
from .base import Type


class TimeRange(Type):
    """TimeRange type

    An object representing a time range between two timestamps.

    Can be used to check if timestamp is within via `in` operator

    Example
    -------
    >>> t0 = datetime.datetime(2018, 1, 1, 14, 00)
    >>> t1 = datetime.datetime(2018, 1, 1, 15, 00)
    >>> time_range = TimeRange(t0, t1)
    >>> test_time = datetime.datetime(2018, 1, 1, 14, 30)
    >>> print(test_time in time_range)
    True
    """

    start_timestamp = Property(datetime.datetime,
                               doc="Start of the time range")
    end_timestamp = Property(datetime.datetime,
                             doc="End of the time range")

    def __init__(self, start_timestamp, end_timestamp, *args, **kwargs):
        if end_timestamp < start_timestamp:
            raise ValueError("start_timestamp must be before end_timestamp")
        super().__init__(start_timestamp, end_timestamp, *args, **kwargs)

    @property
    def duration(self):
        """Duration of the time range"""

        return self.end_timestamp - self.start_timestamp

    def __contains__(self, timestamp):
        """Checks if timestamp is within range

        Parameters
        ----------
        timestamp : datetime.datetime
            Time stamp to check if within range

        Returns
        -------
        bool
            `True` if timestamp within :attr:`start_timestamp` and
            :attr:`end_timestamp` (inclusive)
        """

        return self.start_timestamp <= timestamp <= self.end_timestamp
