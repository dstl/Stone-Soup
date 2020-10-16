# -*- coding: utf-8 -*-
from numbers import Real

import numpy as np
import pytest

from ...types.interval import ClosedContinuousInterval, DisjointIntervals


def test_intervals():
    a = ClosedContinuousInterval(0, 1)
    b = ClosedContinuousInterval(2, 3)
    c = ClosedContinuousInterval(0.5, 2.5)

    assert isinstance(a, ClosedContinuousInterval)

    assert a.left == 0
    assert a.right == 1
    assert a.length == 1

    with pytest.raises(TypeError):
        len(a)

    with pytest.raises(NotImplementedError):
        iter(a)
    with pytest.raises(NotImplementedError):
        reversed(a)

    with pytest.raises(ValueError):
        a.element_iter(1.1)
    elements = []
    for element in a.element_iter(0.2):
        assert isinstance(element, Real)
        elements.append(element)
    assert len(elements) == 6

    with pytest.raises(ValueError):
        a.reversed_element_iter(1.1)
    elements = []
    for element in a.reversed_element_iter(0.2):
        assert isinstance(element, Real)
        elements.append(element)
    assert len(elements) == 6

    with pytest.raises(ValueError):
        a.interval_iter(1.1)
    intervals = []
    for interval in a.interval_iter(0.2):
        assert isinstance(interval, ClosedContinuousInterval)
        assert np.isclose(interval.length, 0.2)
        intervals.append(interval)
    assert len(intervals) == 5

    with pytest.raises(ValueError):
        a.reversed_interval_iter(1.1)
    intervals = []
    for interval in a.reversed_interval_iter(0.2):
        assert isinstance(interval, ClosedContinuousInterval)
        assert np.isclose(interval.length, 0.2)
        intervals.append(interval)
    assert len(intervals) == 5

    assert 0.5 in a
    assert 1.1 not in a

    str_a = str(a)
    assert str_a == '[0, 1]'

    d = a + b

    assert isinstance(d, DisjointIntervals)
    assert np.isclose(d.length, 2)

    e = a + c

    assert isinstance(e, ClosedContinuousInterval)
    assert np.isclose(e.length, 2.5)

    f = b + c

    assert isinstance(f, ClosedContinuousInterval)
    assert np.isclose(f.length, 2.5)

    g = d + c

    assert isinstance(g, ClosedContinuousInterval)
    assert np.isclose(g.length, 3)

    h = a - c
    assert isinstance(h, ClosedContinuousInterval)
    assert np.isclose(h.length, 0.5)

    i = a - ClosedContinuousInterval(0.25, 0.75)
    assert isinstance(i, DisjointIntervals)
    assert len(i) == 2
    assert np.isclose(i.length, 0.5)

    assert not a.check_disjoint(b)
    assert a.check_disjoint(c) == (0, 2.5)

    assert len(d) == 2
    iterate_d = iter(d)
    int1 = next(iterate_d)
    assert int1 == a
    int2 = next(iterate_d)
    assert int2 == b

    reverse_d = reversed(d)
    int1 = next(reverse_d)
    assert int1 == b
    int2 = next(reverse_d)
    assert int2 == a

    assert 0.5 in d
    assert 3.1 not in d
    assert len(d.intervals) == 2

    d.add_interval(c)
    assert len(d) == 1
    assert np.isclose(d.length, 3)

    j = ClosedContinuousInterval(4, 5)
    d.add_interval(j)
    assert len(d) == 2
    assert np.isclose(d.length, 4)

    d.remove_interval(j)
    assert len(d) == 1
    assert np.isclose(d.length, 3)

    k = DisjointIntervals(intervals=[a, c])
    assert len(k) == 2
    assert DisjointIntervals.is_overlap(k.intervals) == (a, c)

    int_list = [a, c]
    merged_int = DisjointIntervals.get_merged_intervals(int_list)
    assert isinstance(merged_int, ClosedContinuousInterval)
    assert np.isclose(merged_int.length, 2.5)

    int_list = [a, c, j]
    merged_int = DisjointIntervals.get_merged_intervals(int_list)
    assert isinstance(merged_int, DisjointIntervals)
    assert len(merged_int) == 2
    assert np.isclose(merged_int.length, 3.5)

    m = DisjointIntervals(intervals=[b, a])
    m.sort()
    assert m.intervals == [a, b]
    m.sort(reverse=True)
    assert m.intervals == [b, a]
