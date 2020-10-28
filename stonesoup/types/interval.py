# -*- coding: utf-8 -*-
import copy
from itertools import combinations
from numbers import Real
from typing import Sequence, Union


class ClosedContinuousInterval:
    """
    Continuous interval class.

    Represents a continuous, closed interval of real numbers.
    Represented by a lower and upper bound.
    """
    def __init__(self, a: Union[int, float], b: Union[int, float]):
        if a == b:
            raise ValueError('a and b must be different')
        elif b < a:
            a, b = b, a
        self._left = a
        self._right = b

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    @property
    def length(self):
        return self.right - self.left

    def __length_hint__(self):
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError('Cannot index a continuous interval')

    def __setitem__(self, key, value):
        raise NotImplementedError('Cannot index a continuous interval')

    def __delitem__(self, key):
        raise NotImplementedError('Cannot index a continuous interval')

    def __missing__(self, key):
        raise NotImplementedError('Cannot index a continuous interval')

    def __iter__(self):
        raise NotImplementedError('Cannot index a continuous interval. Use the self.element_iter()'
                                  ' or self.interval_iter() method')

    def element_iter(self, spacing):
        if spacing > self.length:
            raise ValueError('Spacing must be smaller than, or equal to the size of the interval')
        element_list = []
        value = self.left
        while value <= self.right:
            element_list.append(value)
            value += spacing
        return iter(element_list)

    def interval_iter(self, interval_size):
        if interval_size > self.length:
            raise ValueError('Sub-interval size must be smaller than, or equal to the size of the'
                             ' interval')
        interval_list = []
        current_interval = ClosedContinuousInterval(self.left, self.left + interval_size)
        while current_interval.right <= self.right:
            interval_list.append(current_interval)
            current_interval = ClosedContinuousInterval(current_interval.right,
                                                        current_interval.right + interval_size)
        return iter(interval_list)

    def __reversed__(self):
        raise NotImplementedError('Cannot index a continuous interval. Use the'
                                  'self.reversed_element_iter() or self.reversed_interval_iter()'
                                  'method')

    def reversed_element_iter(self, spacing):
        if spacing > self.length:
            raise ValueError('Spacing must be smaller than, or equal to the size of the interval')
        element_list = []
        value = self.right
        while value >= self.left:
            element_list.append(value)
            value -= spacing
        return iter(element_list)

    def reversed_interval_iter(self, interval_size):
        if interval_size > self.length:
            raise ValueError('Sub-interval size must be smaller than, or equal to the size of the'
                             ' interval')
        interval_list = []
        current_interval = ClosedContinuousInterval(self.right - interval_size, self.right)
        while current_interval.left >= self.left:
            interval_list.append(current_interval)
            current_interval = ClosedContinuousInterval(current_interval.left - interval_size,
                                                        current_interval.left)
        return iter(interval_list)

    def __contains__(self, item):
        if isinstance(item, Real):
            return self.left <= item <= self.right
        else:
            return False

    def __str__(self):
        return '[' + str(self.left) + ', ' + str(self.right) + ']'

    def __add__(self, other):
        if isinstance(other, ClosedContinuousInterval):
            disjoint = self.check_disjoint(other)
            if disjoint:
                return ClosedContinuousInterval(disjoint[0], disjoint[1])
            else:
                return DisjointIntervals([self, other])
        elif isinstance(other, DisjointIntervals):
            return other + self
        else:
            raise NotImplementedError

    def __sub__(self, other):
        if isinstance(other, ClosedContinuousInterval):
            if not self.check_disjoint(other):
                return copy.copy(self)
            else:
                if other.left not in self and other.right not in self:
                    if other.right < self.left or other.left > self.right:
                        return copy.copy(self)
                    else:
                        return None
                elif other.left in self and other.right not in self:
                    if self.left == other.left:
                        return None
                    else:
                        return ClosedContinuousInterval(self.left, other.left)
                elif other.left not in self and other.right in self:
                    if self.right == other.right:
                        return None
                    else:
                        return ClosedContinuousInterval(other.right, self.right)
                else:
                    intervals = []
                    try:
                        int1 = ClosedContinuousInterval(self.left, other.left)
                    except ValueError:
                        pass
                    else:
                        intervals.append(int1)
                    try:
                        int2 = ClosedContinuousInterval(other.right, self.right)
                    except ValueError:
                        pass
                    else:
                        intervals.append(int2)
                    if len(intervals) == 0:
                        return None
                    elif len(intervals) == 1:
                        return intervals[0]
                    else:
                        return DisjointIntervals(intervals)
        elif isinstance(other, DisjointIntervals):
            new_interval = copy.copy(self)
            for interval in other.intervals:
                new_interval -= interval
            return new_interval

    def check_disjoint(self, other):
        """
        Check whether two intervals are disjoint (ie. have no overlap).
        If they overlap, returns new lower and upper bound if the two intervals were to be added.
        Returns False if they do not overlap.
        """
        a_diff = self.right - self.left
        b_diff = other.right - other.left
        lb = min(self.left, other.left)
        ub = max(self.right, other.right)

        if ub - lb < a_diff + b_diff:
            return lb, ub
        else:
            return False


class DisjointIntervals:
    """
    Disjoint closed continuous intervals class.

    Represents a set of continuous, closed intervals of real numbers.
    Represented by a list of ClosedContinuousInterval types.
    """

    def __init__(self, intervals: Sequence[ClosedContinuousInterval]):
        self._intervals = intervals

    def __str__(self):
        return str([str(interval) for interval in self.intervals])

    def __len__(self):
        return len(self.intervals)

    @property
    def length(self):
        return sum(interval.length for interval in self.intervals)

    def __length_hint__(self):
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError('Cannot index continuous intervals')

    def __setitem__(self, key, value):
        raise NotImplementedError('Cannot index continuous intervals')

    def __delitem__(self, key):
        raise NotImplementedError('Cannot index continuous intervals')

    def __missing__(self, key):
        raise NotImplementedError('Cannot index continuous intervals')

    def _iter(self, reverse):
        return iter(sorted(self.intervals, key=lambda interval: interval.left, reverse=reverse))

    def __iter__(self):
        return self._iter(reverse=False)

    def __reversed__(self):
        return self._iter(reverse=True)

    def __contains__(self, item):
        return any(item in interval for interval in self.intervals)

    def __add__(self, other):
        if isinstance(other, DisjointIntervals):
            return DisjointIntervals.get_merged_intervals(self.intervals + other.intervals)
        elif isinstance(other, ClosedContinuousInterval):
            return DisjointIntervals.get_merged_intervals(self.intervals + [other])
        else:
            raise NotImplementedError

    def __sub__(self, other):
        if isinstance(other, DisjointIntervals):
            new_intervals = copy.copy(self.intervals)
            for other_interval in other.intervals:
                for i in range(len(new_intervals)):
                    if new_intervals[i].check_disjoint(other_interval):
                        new_intervals[i] = new_intervals[i] - other_interval
                new_intervals = [interval for interval in new_intervals if interval is not None]
            if len(new_intervals) == 0:
                return None
            else:
                return DisjointIntervals.get_merged_intervals(new_intervals)
        elif isinstance(other, ClosedContinuousInterval):
            new_intervals = [interval - other for interval in copy.copy(self.intervals)]
            new_intervals = [interval for interval in new_intervals if interval is not None]
            if len(new_intervals) == 0:
                return None
            else:
                return DisjointIntervals.get_merged_intervals(new_intervals)
        else:
            raise NotImplementedError

    @property
    def intervals(self):
        return self._intervals

    def add_interval(self, interval):
        if isinstance(interval, ClosedContinuousInterval):
            self._intervals.append(interval)
            self.merge_overlaps()
        else:
            raise ValueError('interval must be a ClosedContinuousInterval type')

    def add_intervals(self, intervals):
        if isinstance(intervals, DisjointIntervals):
            self._intervals.extend(intervals.intervals)
            self.merge_overlaps()
        else:
            raise ValueError('intervals must be a DisjointIntervals type')

    def remove_interval(self, interval):
        if interval not in self.intervals:
            raise ValueError('Interval not in list')
        else:
            self.intervals.remove(interval)

    @staticmethod
    def is_overlap(intervals):
        for int_1, int_2 in combinations(intervals, 2):
            if int_1.check_disjoint(int_2):
                return int_1, int_2
        return None

    # TODO: Check if only one interval after merge, then convert to CloseContinuousInterval type
    def merge_overlaps(self):
        """Merge intervals in-place"""
        while True:
            try:
                int_1, int_2 = DisjointIntervals.is_overlap(self.intervals)
            except TypeError:
                break
            else:
                self.remove_interval(int_1)
                self.remove_interval(int_2)
                new_interval = int_1 + int_2
                if isinstance(new_interval, ClosedContinuousInterval):
                    self.add_interval(new_interval)
                else:
                    self.add_intervals(new_interval)

    @staticmethod
    def get_merged_intervals(intervals: Sequence[ClosedContinuousInterval]):
        """Merge a list of intervals"""
        new_intervals = copy.copy(intervals)
        while True:
            try:
                int_1, int_2 = DisjointIntervals.is_overlap(new_intervals)
            except TypeError:
                break
            else:
                new_intervals.remove(int_1)
                new_intervals.remove(int_2)
                new_interval = int_1 + int_2
                if isinstance(new_interval, ClosedContinuousInterval):
                    new_intervals.append(new_interval)
                else:
                    new_intervals.extend(new_interval.intervals)
        if len(new_intervals) == 1:
            return new_intervals[0]
        else:
            return DisjointIntervals(intervals=new_intervals)

    def sort(self, reverse=False):
        self.intervals.sort(key=lambda interval: interval.left, reverse=reverse)
