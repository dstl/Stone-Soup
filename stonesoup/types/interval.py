import copy
import operator
from itertools import combinations
from numbers import Real
from typing import Sequence, Union, MutableSequence, Tuple

from ..base import Property
from ..types import Type


class Interval(Type):
    """
    Closed continuous interval class.

    Represents a continuous, closed interval of real numbers.
    Represented by a lower and upper bound.
    """
    start: Union[int, float] = Property(doc="Lower bound of interval")
    end: Union[int, float] = Property(doc="Upper bound of interval")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.start >= self.end:
            raise ValueError('Must have left < right')

    def __hash__(self):
        return hash((self.start, self.end))

    @property
    def left(self):
        return self.start

    @property
    def right(self):
        return self.end

    @property
    def length(self):
        return self.end - self.start

    def __contains__(self, item):
        if isinstance(item, Real):
            return self.start <= item <= self.end
        elif isinstance(item, Interval):
            return self & item == item
        else:
            return False

    def __str__(self):
        return '[{left}, {right}]'.format(left=self.start, right=self.end)

    # def __repr__(self):
    #     return 'Interval{interval}'.format(interval=str(self))

    def __eq__(self, other):
        return isinstance(other, Interval) and (self.start, self.end) == (other.start, other.end)

    def __and__(self, other):
        """Set-like intersection"""

        if not isinstance(other, Interval):
            raise ValueError("Can only intersect with Interval types")

        if not self.isdisjoint(other):
            new_interval = max(self.start, other.start), min(self.end, other.end)
            if new_interval[0] == new_interval[1]:
                return None
            else:
                return type(self)(new_interval[0], new_interval[1])
        else:
            return None

    def __or__(self, other):
        """Set-like union"""

        if not isinstance(other, Interval):
            raise ValueError("Can only union with Interval types")

        if not self.isdisjoint(other):
            return [type(self)(min(self.start, other.start), max(self.end, other.end))]
        else:
            return [copy.copy(self), copy.copy(other)]

    def __sub__(self, other):
        """Set-like difference"""

        if other is None:
            return [copy.copy(self)]
        elif not isinstance(other, Interval):
            raise ValueError("Can only subtract Interval types from Interval types")
        elif self.isdisjoint(other):
            return [copy.copy(self)]
        elif other.start <= self.start and self.end <= other.end:
            return [None]
        elif other.start <= self.start:
            return [Interval(other.end, self.end)]
        elif self.end <= other.end:
            return [Interval(self.start, other.start)]
        else:
            return [Interval(self.start, other.start), Interval(other.end, self.end)]

    def __xor__(self, other):
        """Set-like symmetric difference"""

        if not isinstance(other, Interval):
            raise ValueError("Can only subtract Interval types from Interval types")

        if self.isdisjoint(other):
            return [copy.copy(self), copy.copy(other)]

        # Union will return list with one interval
        return (self | other)[0] - (self & other)

    def __le__(self, other):
        """Subset check"""

        if not isinstance(other, Interval):
            raise ValueError("Can only compare Interval types to Interval types")

        return self in other

    def __lt__(self, other):
        """Proper subset check"""

        if not isinstance(other, Interval):
            raise ValueError("Can only compare Interval types to Interval types")

        return other.start < self.start and self.end < other.end

    def __ge__(self, other):
        """Superset check"""

        if not isinstance(other, Interval):
            raise ValueError("Can only compare Interval types to Interval types")

        return other <= self

    def __gt__(self, other):
        """Proper superset check"""

        if not isinstance(other, Interval):
            raise ValueError("Can only compare Interval types to Interval types")

        return other < self

    def isdisjoint(self, other):
        """
        Check whether two intervals are disjoint (do not overlap).
        Returns True if they are disjoint.
        Note: returns False if intervals endpoints 'meet'. For example [0, 1] meets [1, 2].
        """
        if not isinstance(other, Interval):
            raise ValueError("Interval types can only overlap with Interval types")

        max_start = max(self.start, other.start)
        min_end = min(self.end, other.end)

        return max_start > min_end


class Intervals(Type):
    """
    Disjoint closed continuous intervals class.

    Represents a set of continuous, closed intervals of real numbers. Represented by a list of
    :class:`Interval` types.
    """

    intervals: MutableSequence[Interval] = Property(
        default=None,
        doc="Container of :class:`Interval`")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.intervals is None:
            self.intervals = list()
        elif not isinstance(self.intervals, MutableSequence):
            if isinstance(self.intervals, (Interval, Tuple)):
                self.intervals = [self.intervals]
            else:
                raise TypeError("Must contain Interval types")
        elif len(self.intervals) == 2 and all(isinstance(elem, Real) for elem in self.intervals):
            self.intervals = [self.intervals]

        for i in range(len(self.intervals)):
            interval = self.intervals[i]
            if not isinstance(interval, Interval):
                if isinstance(interval, Sequence) and len(interval) == 2:
                    self.intervals[i] = Interval(*interval)
                else:
                    raise TypeError("Individual intervals must be an Interval or Sequence type")

        self.intervals = self.get_merged_intervals(self.intervals)

    def __hash__(self):
        return hash(tuple(self.intervals))

    @staticmethod
    def overlap(intervals):
        """
        Determine whether a pair of intervals in a list overlap (are not disjoint).
        Returns a pair of overlapping intervals if there are any, otherwise returns None.
        """
        for int_1, int_2 in combinations(intervals, 2):
            if not int_1.isdisjoint(int_2):
                return int_1, int_2
        return None

    def isdisjoint(self, other):
        """
        Determine whether all intervals in an :class:`Intervals` type are disjoint from all those
        in another.
        """

        if not isinstance(other, Intervals):
            raise ValueError("Can only compare Intervals to Intervals")
        return all(interval.isdisjoint(other_int) for other_int in other for interval in self)

    @staticmethod
    def get_merged_intervals(intervals):
        """Merge all intervals. Ie. combine any intervals that overlap, returning a new list of
        disjoint intervals."""

        new_intervals = copy.copy(intervals)

        while True:
            # Continue while there are overlaps
            try:
                int_1, int_2 = Intervals.overlap(new_intervals)
            except TypeError:
                break
            else:
                # Remove overlapping intervals and add their union
                new_intervals.remove(int_1)
                new_intervals.remove(int_2)
                new_intervals.extend(int_1 | int_2)

        return sorted(new_intervals, key=operator.attrgetter('left'))

    def __contains__(self, item):

        if not isinstance(item, (Real, Interval, Intervals)):
            return False

        if isinstance(item, Intervals):
            return all(any(other_int in interval for interval in self) for other_int in item)

        return any(item in interval for interval in self)

    def __str__(self):
        return str([[interval.start, interval.end] for interval in self])

    @property
    def length(self):
        return sum(interval.length for interval in self)

    def _iter(self, reverse):
        for interval in sorted(self.intervals, key=operator.attrgetter('left'), reverse=reverse):
            yield interval

    def __iter__(self):
        return self._iter(reverse=False)

    def __reversed__(self):
        return self._iter(reverse=True)

    def __eq__(self, other):
        if len(self) == 0:
            return True if len(other) == 0 else False
        if isinstance(other, Interval):
            other = Intervals(other)

        return isinstance(other, Intervals) and all(
            any(int1 == int2 for int2 in other) for int1 in self)

    def __and__(self, other):
        """Set-like intersection"""

        if other is None:
            return False
        if not isinstance(other, (Interval, Intervals)):
            raise ValueError("Can only intersect with Intervals types")

        if isinstance(other, Interval):
            other = Intervals(other)

        new_intervals = list()
        for interval in self:
            for other_interval in other:
                new_interval = interval & other_interval
                if new_interval:
                    new_intervals.append(new_interval)
        new_intervals = self.get_merged_intervals(new_intervals)
        new_intervals = type(self)(new_intervals)
        return new_intervals

    def __or__(self, other):
        """Set-like union"""
        if not isinstance(other, (Interval, Intervals)):
            raise ValueError('Can only union with Intervals types')

        if isinstance(other, Interval):
            other = Intervals(other)

        new_intervals = self.intervals + other.intervals
        new_intervals = self.get_merged_intervals(new_intervals)
        new_intervals = type(self)(new_intervals)
        return new_intervals

    def __sub__(self, other):
        """Set-like difference"""

        if other is None:
            return self.copy()
        elif not isinstance(other, (Interval, Intervals)):
            raise ValueError("Can only subtract Intervals from Intervals")

        if isinstance(other, Interval):
            other = Intervals(other)

        new_intervals = copy.copy(self.intervals)

        for other_interval in other:
            temp_intervals = list()

            for interval in new_intervals:
                diff = interval - other_interval
                if diff[0] is not None:
                    temp_intervals.extend(diff)
            new_intervals = temp_intervals
        new_intervals = type(self)(new_intervals)
        return new_intervals

    def __xor__(self, other):
        """Set-like symmetric-difference"""

        if not isinstance(other, (Interval, Intervals)):
            raise ValueError("Can only compare Intervals from Intervals")

        if isinstance(other, Interval):
            other = type(self)(other)

        return (self | other) - (self & other)

    def __le__(self, other):
        """Subset check"""

        if not isinstance(other, (Interval, Intervals)):
            raise ValueError("Can only compare Intervals to Intervals")

        if isinstance(other, Interval):
            other = Intervals(other)

        return all(any(interval <= other_int for other_int in other) for interval in self)

    def __lt__(self, other):
        """"Proper subset check"""

        if not isinstance(other, (Interval, Intervals)):
            raise ValueError("Can only compare Intervals to Intervals")

        if isinstance(other, Interval):
            other = Intervals(other)

        return all(any(interval < other_int for other_int in other) for interval in self)

    def __ge__(self, other):
        """Superset check"""

        if not isinstance(other, (Interval, Intervals)):
            raise ValueError("Can only compare Intervals to Intervals")

        if isinstance(other, Interval):
            other = Intervals(other)

        return other <= self

    def __gt__(self, other):
        """Proper superset check"""

        if not isinstance(other, (Interval, Intervals)):
            raise ValueError("Can only compare Intervals to Intervals")

        if isinstance(other, Interval):
            other = Intervals(other)

        return other < self

    def copy(self):
        return Intervals(copy.copy(self.intervals))

    def __len__(self):
        return len(self.intervals)

    def remove(self, elem):
        if not isinstance(elem, Interval):
            raise ValueError("Intervals only contain Interval types")
        try:
            self.intervals.remove(elem)
        except ValueError:
            raise ValueError("Interval not in list")

    def discard(self, elem):
        try:
            self.remove(elem)
        except ValueError:
            pass

    def pop(self):
        if len(self) == 0:
            raise KeyError("Contains no intervals")

        interval = next(iter(self))
        self.intervals.remove(interval)

        return interval

    def clear(self):
        self.intervals = list()
