import datetime
import copy
from itertools import combinations, permutations

from ..base import Property
from ..types.interval import Interval, Intervals


class TimeRange(Interval):
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

    start: datetime.datetime = Property(doc="Start of the time range")
    end: datetime.datetime = Property(doc="End of the time range")

    @property
    def start_timestamp(self):
        return self.start

    @property
    def end_timestamp(self):
        return self.end

    @property
    def duration(self):
        """Duration of the time range"""

        return self.length

    @property
    def key_times(self):
        """Times the TimeRange begins and ends"""
        return [self.start, self.end]

    def __contains__(self, time):
        """Checks if timestamp is within range

        Parameters
        ----------
        time : Union[datetime.datetime, TimeRange]
            Time stamp or range to check if within range

        Returns
        -------
        bool
            `True` if timestamp within :attr:`start` and
            :attr:`end` (inclusive)
        """
        if isinstance(time, datetime.datetime):
            return self.start <= time <= self.end
        else:
            return super().__contains__(time)

    def __eq__(self, other):
        return isinstance(other, TimeRange) and super().__eq__(other)

    def __sub__(self, time_range):
        """Removes the overlap between this instance and another :class:`~.TimeRange`, or
        :class:`~.CompoundTimeRange`.

        Parameters
        ----------
        time_range: Union[TimeRange, CompoundTimeRange]

        Returns
        -------
        TimeRange
            This instance less the overlap with the other time_range
        """
        if time_range is None:
            return copy.copy(self)
        if not isinstance(time_range, TimeRange) and not isinstance(time_range, CompoundTimeRange):
            raise TypeError("Supplied parameter must be a TimeRange or CompoundTimeRange object")
        if isinstance(time_range, CompoundTimeRange):
            ans = self
            for t_range in time_range.time_ranges:
                ans -= t_range
                if not ans:
                    return None
            return ans
        else:
            overlap = self & time_range
            if overlap is None:
                return self
            if self == overlap:
                return None
            if self.start < overlap.start:
                start = self.start
            else:
                start = overlap.end
            if self.end > overlap.end:
                end = self.end
            else:
                end = overlap.start
            if self.start < overlap.start and \
               self.end > overlap.end:
                return CompoundTimeRange([TimeRange(self.start, overlap.start),
                                         TimeRange(overlap.end, self.end)])
            else:
                return TimeRange(start, end)

    def __and__(self, time_range):
        """Finds the intersection between this instance and another :class:`~.TimeRange` or
        :class:`.~CompoundTimeRange`

        Parameters
        ----------
        time_range: Union[TimeRange, CompoundTimeRange]

        Returns
        -------
        TimeRange
            The times contained by both this and `time_range`
        """
        if time_range is None:
            return None
        if isinstance(time_range, CompoundTimeRange):
            return time_range & self
        if not isinstance(time_range, TimeRange):
            raise TypeError("Supplied parameter must be a TimeRange object")
        return super().__and__(time_range)

    def __or__(self, other):
        return super().__or__(other)


class CompoundTimeRange(Intervals):
    """CompoundTimeRange type

    A container class representing one or more :class:`~.TimeRange` objects together
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.time_ranges, list):
            raise TypeError("Time_ranges must be a list")
        for component in self.time_ranges:
            if not isinstance(component, TimeRange):
                raise TypeError("Time_ranges must contain only TimeRange objects")
        self._remove_overlap()
        self._fuse_components()

    @property
    def time_ranges(self):
        return self.intervals

    @property
    def duration(self):
        """Duration of the time range"""
        if len(self.time_ranges) == 0:
            return datetime.timedelta(0)
        total_duration = datetime.timedelta(0)
        for component in self.time_ranges:
            total_duration = total_duration + component.duration
        return total_duration

    @property
    def key_times(self):
        """Returns all timestamps at which a component starts or ends"""
        key_times = set()
        for component in self.time_ranges:
            key_times.add(component.start)
            key_times.add(component.end)
        return sorted(key_times)

    def _remove_overlap(self):
        """Removes overlap between components of `time_ranges`"""
        if len(self.time_ranges) in {0, 1}:
            return
        if all([component & component2 is None for (component, component2) in
                combinations(self.time_ranges, 2)]):
            return
        overlap_check = CompoundTimeRange()
        for time_range in self.time_ranges:
            if time_range - overlap_check:
                overlap_check.add(time_range - overlap_check & time_range)
        self.intervals = copy.copy(overlap_check.time_ranges)

    def _fuse_components(self):
        """Fuses two time ranges [a,b], [b,c] into [a,c] for all such pairs in this instance"""
        for (component, component2) in permutations(self.time_ranges, 2):
            if component.end == component2.start:
                fused_component = TimeRange(component.start, component2.end)
                self.remove(component)
                self.remove(component2)
                self.add(fused_component)
                # To avoid issues with having removed objects from the permutations
                self._fuse_components()

    def add(self, time_range):
        """Add a :class:`~.TimeRange` or :class:`~.CompoundTimeRange` object to `time_ranges`."""
        if time_range is None:
            return
        if isinstance(time_range, CompoundTimeRange):
            for component in time_range.time_ranges:
                self.add(component)
        elif isinstance(time_range, TimeRange):
            self.time_ranges.append(time_range)
        else:
            raise TypeError("Supplied parameter must be a TimeRange or CompoundTimeRange object")
        self._remove_overlap()
        self._fuse_components()

    def remove(self, time_range):
        """Removes a :class:`.~TimeRange` object from the time ranges.
        It must be a member of self.time_ranges"""
        if not isinstance(time_range, TimeRange):
            raise TypeError("Supplied parameter must be a TimeRange object")
        if time_range in self.time_ranges:
            self.time_ranges.remove(time_range)
        elif time_range in self:
            for component in self.time_ranges:
                if time_range in component:
                    new = component - time_range
                    self.time_ranges.remove(component)
                    self.add(new)
        else:
            raise ValueError("Supplied parameter must be a member of time_ranges")

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

        if isinstance(time, datetime.datetime):
            for component in self.time_ranges:
                if time in component:
                    return True
            return False
        elif isinstance(time, (TimeRange, CompoundTimeRange)):
            return super().__contains__(time)
        else:
            raise TypeError("Supplied parameter must be an instance of either "
                            "datetime, TimeRange, or CompoundTimeRange")

    def __eq__(self, other):
        return isinstance(other, CompoundTimeRange) and super().__eq__(other)

    def __sub__(self, time_range):
        """Removes any overlap between this and another :class:`~.TimeRange` or
        :class:`.~CompoundTimeRange` from this instance

        Parameters
        ----------
        time_range: Union[TimeRange, CompoundTimeRange]

        Returns
        -------
        CompoundTimeRange
            The times contained by this but not time_range.  May be empty.
        """
        if time_range is None:
            return copy.copy(self)
        ans = CompoundTimeRange()
        for component in self.time_ranges:
            ans.add(component - time_range)
        return ans

    def __and__(self, time_range):
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
        if time_range is None:
            return None
        if not isinstance(time_range, (TimeRange, CompoundTimeRange)):
            raise TypeError("Supplied parameter must be an instance of either "
                            "TimeRange, or CompoundTimeRange")
        else:
            return super().__and__(time_range)
