import datetime
from typing import Union
from itertools import combinations

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

    def minus(self, time_range):
        """Removes the overlap between this instance and another :class:`~.TimeRange`, or
        :class:`~.CompoundTimeRange`.

        Parameters
        ----------
        time_range: Union[TimeRange, CompoundTimeRange]

        Returns
        -------
        TimeRange
            This instance less the overlap
        """
        if isinstance(time_range, CompoundTimeRange):
            ans = self
            for t_range in time_range.time_ranges:
                ans = ans.minus(t_range)
                if not ans:
                    return None
            return ans
        else:
            overlap = self.overlap(time_range)
            if self == overlap:
                return None
            if self.start_timestamp < overlap.start_timestamp:
                start = self.start_timestamp
            else:
                start = overlap.end_timestamp
            if self.end_timestamp > overlap.end_timestamp:
                end = self.end_timestamp
            else:
                end = overlap.start_timestamp
            if self.start_timestamp < overlap.start_timestamp and \
               self.end_timestamp > overlap.end_timestamp:
                return CompoundTimeRange(TimeRange(self.start_timestamp, overlap.start_timestamp),
                                         TimeRange(self.end_timestamp, overlap.end_timestamp))
            else:
                return TimeRange(start, end)

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
    time_ranges: list[TimeRange] = Property(doc="List of TimeRange objects.  Can be empty",
                                            default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.time_ranges is None:
            self.time_ranges = []
        if not isinstance(self.time_ranges, list):
            raise TypeError("Time_ranges must be a list")
        for component in self.time_ranges:
            if not isinstance(component, TimeRange):
                raise TypeError("Time_ranges must contain only TimeRange objects")
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
        """Removes overlap between components of time_ranges"""
        if len(self.time_ranges) == 0:
            return
        if all([component.overlap(component2) is None for (component, component2) in
                combinations(self.time_ranges, 2)]):
            return
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

    def minus(self, time_range):
        """Removes any overlap between this and another :class:`~.TimeRange` or
        :class:`.~CompoundTimeRange`

        Parameters
        ----------
        time_range: Union[TimeRange, CompoundTimeRange]

        Returns
        -------
        CompoundTimeRange
            The times contained by both this and time_range
        """
        ans = CompoundTimeRange()
        for component in self.time_ranges:
            ans.add(component.minus(time_range))
        return ans

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
            if total_overlap == CompoundTimeRange():
                return None
            return total_overlap
        elif isinstance(time_range, TimeRange):
            for component in self.time_ranges:
                total_overlap.add(component.overlap(time_range))
            if total_overlap == CompoundTimeRange():
                return None
            return total_overlap
        else:
            raise TypeError("Supplied parameter must be an instance of either "
                            "TimeRange, or CompoundTimeRange")

