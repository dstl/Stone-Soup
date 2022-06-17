import datetime
from typing import Union

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

    start_timestamp: datetime.datetime = Property(doc="Start of the time range")
    end_timestamp: datetime.datetime = Property(doc="End of the time range")

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

    def overlap(self, time_range):
        """Finds the intersection between this instance and another :class:`~.TimeRange`

        Parameters
        ----------
        time_range: TimeRange

        Returns
        -------
        TimeRange
            The times contained by both this and time_range
        """
        start_timestamp = max(self.start_timestamp, time_range.start_timestamp)
        end_timestamp = min(self.end_timestamp, time_range.end_timestamp)
        if end_timestamp > start_timestamp:
            return TimeRange(start_timestamp, end_timestamp)
        else:
            return None


class CompoundTimeRange(Type):
    """CompoundTimeRange type

    A container class representing one or more :class:`TimeRange` objects together
    """
    time_ranges: list[TimeRange] = Property(doc="List of TimeRange objects", default=None)

    def __init__(self, time_ranges, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not time_ranges:
            self.time_ranges = []
        self.remove_overlap()

    @property
    def duration(self):
        """Duration of the time range"""
        total_duration = 0
        for component in self.time_ranges:
            total_duration += component.duration

    @property
    def key_times(self):
        """Returns all timestamps at which a component starts or ends"""
        key_times = set()
        for component in self.time_ranges:
            key_times.add(component.start_timestamp)
            key_times.add(component.end_timestamp)
        return list(key_times).sort()

    def remove_overlap(self):
        """Returns a :class:`~.CompoundTimeRange` with overlap removed"""
        overlap_check = CompoundTimeRange()
        for time_range in self.time_ranges:
            overlap_check.add(time_range.minus(overlap_check.overlap(time_range)))
        self.time_ranges = overlap_check

    def add(self, time_range):
        """Add a :class:`~.TimeRange` or :class:`~.CompoundTimeRange` object to `time_ranges`"""
        if time_range is None:
            return
        if isinstance(time_range, CompoundTimeRange):
            for component in time_range.time_ranges:
                self.add(component)
        else:
            self.time_ranges.append(time_range)
        self.remove_overlap()

    def __contains__(self, time):
        """Checks if timestamp or is within range

        Parameters
        ----------
        time : Union[datetime.datetime, TimeRange, CompoundTimeRange]
            Time stamp or range to check if contained within this instance

        Returns
        -------
        bool
            `True` if time is fully contained within this instance
        """

        if isinstance(time, datetime):
            for component in self.time_ranges:
                if datetime in component:
                    return True
            return False
        elif isinstance(time, TimeRange) or isinstance(time, CompoundTimeRange):
            return True if self.overlap(time) == time else False
        else:
            raise TypeError("Supplied parameter must be an instance of either "
                            "datetime, TimeRange, or CompoundTimeRange")

    def overlap(self, time_range):
        """Finds the intersection between this instance and another time range

        In the case of an input :class:`~.CompoundTimeRange` this  is done recursively.

        Parameters
        ----------
        time_range: Union[TimeRange, CompoundTimeRange]

        Returns
        -------
        CompoundTimeRange
            The times contained by both this and time_range
        """
        total_overlap = CompoundTimeRange()
        if isinstance(time_range, CompoundTimeRange):
            for component in time_range.time_ranges:
                total_overlap.add(self.overlap(component))
            return total_overlap
        elif isinstance(time_range, TimeRange):
            for component in self.time_ranges:
                total_overlap.add(component.overlap(time_range))
            return total_overlap
        else:
            raise TypeError("Supplied parameter must be an instance of either "
                            "TimeRange, or CompoundTimeRange")

