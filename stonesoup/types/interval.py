# -*- coding: utf-8 -*-
import copy
import operator
from itertools import combinations
from numbers import Real
from typing import Sequence, Union, MutableSequence

from stonesoup.base import Property
from stonesoup.types import Type


class Interval(Type):
    left: Union[int, float] = Property(doc="Lower bound of interval")
    right: Union[int, float] = Property(doc="Upper bound of interval")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.left > self.right:
            raise ValueError('left > right')

    @property
    def length(self):
        return self.right - self.left

    def __contains__(self, item):
        if isinstance(item, Real):
            return self.left <= item <= self.right
        elif isinstance(item, Interval):
            return self & item == item
        else:
            return False

    def __str__(self):
        return '[{left}, {right}]'.format(left=self.left, right=self.right)

    def __repr__(self):
        return 'Interval{interval}'.format(interval=str(self))

    def __eq__(self, other):
        return isinstance(other, Interval) and (self.left, self.right) == (other.left, other.right)

    def __and__(self, other):
        if isinstance(other, Interval):
            if self.overlap(other):
                new_interval = (max(self.left, other.left), min(self.right, other.right))
                if new_interval[0] == new_interval[1]:
                    return None
                else:
                    return Interval(*new_interval)
            else:
                return None
        else:
            raise ValueError("Can only intersect with Interval types")

    def __or__(self, other):
        if isinstance(other, Interval):
            if self.overlap(other):
                return [Interval(min(self.left, other.left), max(self.right, other.right))]
            else:
                return copy.copy(self), copy.copy(other)
        else:
            raise ValueError("Can only union with Interval types")

    def __sub__(self, other):

        if not isinstance(other, Interval):
            raise ValueError("Can only subtract Interval types from Interval types")
        elif not self.overlap(other):
            return copy.copy(self)
        elif other.left <= self.left and self.right <= other.right:
            return None
        elif other.left <= self.left:
            return Interval(other.right, self.right)
        elif self.right <= other.right:
            return Interval(self.left, other.left)
        else:
            return Interval(self.left, other.left), Interval(other.right, self.right)

    def overlap(self, other):
        """
        Check whether two intervals overlap (are not disjoint).
        Returns False if they do not overlap.
        Note: returns True if intervals endpoints 'meet'. For example [0, 1] meets [1, 2].
        """
        if not isinstance(other, Interval):
            raise ValueError("Interval types can only overlap with Interval types")

        lb = min(self.left, other.left)
        ub = max(self.right, other.right)

        return (ub - lb < self.length + other.length) \
               or (self.right == other.left) or (other.right == self.left)


class Intervals(Type):
    """
    Disjoint closed continuous intervals class.

    Represents a set of continuous, closed intervals of real numbers. Represented by a list of
    ContinuousInterval types.
    """

    intervals: MutableSequence[Interval] = Property(
        default=None,
        doc="Container of :class:`Interval`")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.intervals is None:
            self.intervals = list()

        if not isinstance(self.intervals, MutableSequence):
            if isinstance(self.intervals, Interval):
                self.intervals = [self.intervals]
            else:
                raise ValueError("Must contain Interval types")
        elif len(self.intervals) == 2 and all(isinstance(elem, Real) for elem in self.intervals):
            self.intervals = [self.intervals]

        for i in range(len(self.intervals)):
            interval = self.intervals[i]
            if not isinstance(interval, Interval):
                if isinstance(interval, Sequence) and len(interval) == 2:
                    self.intervals[i] = Interval(*interval)
                else:
                    raise ValueError("Individual intervals must be an Interval or Sequence type")

        self.intervals = self.get_merged_intervals(self.intervals)

    @staticmethod
    def overlap(intervals):
        for int_1, int_2 in combinations(intervals, 2):
            if int_1.overlap(int_2):
                return int_1, int_2
        return None

    @staticmethod
    def meet(intervals):
        for int_1, int_2 in combinations(intervals, 2):
            if int_1.meet(int_2):
                return int_1, int_2
        return None

    @staticmethod
    def get_merged_intervals(intervals):
        """Merge intervals"""

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
        if not isinstance(item, (Real, Interval)):
            return False
        return any(item in interval for interval in self)

    def __str__(self):
        return str([[interval.left, interval.right] for interval in self.intervals])

    def __repr__(self):
        return 'Intervals{intervals}'.format(intervals=str(self))

    def __len__(self):
        return len(self.intervals)

    @property
    def length(self):
        return sum(interval.length for interval in self.intervals)

    def _iter(self, reverse):
        for interval in sorted(self.intervals, key=operator.attrgetter('left'), reverse=reverse):
            yield interval

    def __iter__(self):
        return self._iter(reverse=False)

    def __reversed__(self):
        return self._iter(reverse=True)

    def __eq__(self, other):
        return isinstance(other, Intervals) and \
               all(any(int1 == int2 for int2 in other) for int1 in self)

    def __or__(self, other):

        if not isinstance(other, Intervals):
            raise ValueError('Can only union with Intervals types')

        new_intervals = self.intervals + other.intervals
        new_intervals = self.get_merged_intervals(new_intervals)
        new_intervals = Intervals(new_intervals)
        return new_intervals

    def __and__(self, other):

        if not isinstance(other, Intervals):
            raise ValueError('Can only intersect with Intervals types')

        new_intervals = list()
        for interval in self.intervals:
            for other_interval in other.intervals:
                new_interval = interval & other_interval
                if new_interval:
                    new_intervals.append(new_interval)
        new_intervals = self.get_merged_intervals(new_intervals)
        new_intervals = Intervals(new_intervals)
        return new_intervals

    def remove_interval(self, interval):
        if interval not in self.intervals:
            raise ValueError('Interval not in list')
        else:
            self.intervals.remove(interval)

def test_int():
    a = Interval(0, 1)
    b = Interval(0.5, 1.5)
    c = Interval(2, 3)
    I = Intervals([a, b, c])

    assert a|b == [Interval(0, 1.5)]
    assert a&b == Interval(0.5, 1)
    assert a|c == [a, c]
    assert a&c is None

    J = Intervals([(1.5, 6)])

    print(a - b)
    assert False