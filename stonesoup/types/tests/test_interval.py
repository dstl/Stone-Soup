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
        next(a.element_iter(1.1))

    elements = []
    a_iter = a.element_iter(0.2)
    for element in a_iter:
        assert isinstance(element, Real)
        elements.append(element)
    assert len(elements) == 6

    with pytest.raises(ValueError):
        next(a.reversed_element_iter(1.1))
    elements = []
    a_iter = a.reversed_element_iter(0.2)
    for element in a_iter:
        assert isinstance(element, Real)
        elements.append(element)
    assert len(elements) == 6

    with pytest.raises(ValueError):
        next(a.interval_iter(1.1))
    intervals = []
    a_iter = a.interval_iter(0.2)
    for interval in a_iter:
        assert isinstance(interval, ClosedContinuousInterval)
        assert np.isclose(interval.length, 0.2)
        intervals.append(interval)
    assert len(intervals) == 5

    with pytest.raises(ValueError):
        next(a.reversed_interval_iter(1.1))
    intervals = []
    a_iter = a.reversed_interval_iter(0.2)
    for interval in a_iter:
        assert isinstance(interval, ClosedContinuousInterval)
        assert np.isclose(interval.length, 0.2)
        intervals.append(interval)
    assert len(intervals) == 5

    assert 0.5 in a
    assert 1.1 not in a
    assert ClosedContinuousInterval(0.1, 0.2) in a
    assert ClosedContinuousInterval(-0.1, 0.2)not in a
    assert ClosedContinuousInterval(0.1, 1.1) not in a
    assert b not in a

    str_a = str(a)
    assert str_a == '[0, 1]'

    repr_a = a.__repr__()
    assert repr_a == 'ClosedContinuousInterval[0, 1]'

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

    g = c + d

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

    with pytest.raises(ValueError, match="intervals must be a DisjointIntervals type"):
        d.add_intervals(c)

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

    n = ClosedContinuousInterval(2, 1)
    assert n.left == 1
    assert n.right == 2

    with pytest.raises(ValueError, match="left and right values must be different"):
        ClosedContinuousInterval(1, 1)

    d = a + b
    e = a + j
    q = d + e
    assert isinstance(q, DisjointIntervals)
    assert len(q) == 3
    assert np.isclose(q.length, 3)

    assert 'a string' not in a

    r = d - e
    assert isinstance(r, ClosedContinuousInterval)
    assert np.isclose(r.length, 1)

    s = d - c

    assert isinstance(s, DisjointIntervals)
    assert len(s) == 2
    assert np.isclose(s.length, 1)

    t = ClosedContinuousInterval(0, 3)
    d = a + b
    u = d - t

    assert u is None

    v = DisjointIntervals(intervals=[ClosedContinuousInterval(0, 1.5),
                                     ClosedContinuousInterval(1.9, 3)])

    w = d - v
    assert w is None

    d = a + b

    d.add_intervals(DisjointIntervals(intervals=[j, t]))
    assert len(d) == 2
    assert np.isclose(d.length, 4)

    d = a + b

    with pytest.raises(ValueError, match="intervals must be a DisjointIntervals type"):
        d.add_intervals(c)

    with pytest.raises(ValueError):
        d.add_interval(None)

    with pytest.raises(ValueError):
        d.remove_interval(c)

    assert str(d) == str([str(a), str(b)])

    assert d.__repr__() == 'DisjointIntervals{intervals}'.format(intervals=str(d))

    assert a - b == a

    assert a - ClosedContinuousInterval(-1, 2) is None

    assert b - b is None

    e = b - d

    assert e is None

    with pytest.raises(NotImplementedError):
        b - None

    e = b - DisjointIntervals([ClosedContinuousInterval(2, 2.4), ClosedContinuousInterval(2.6, 3)])
    assert isinstance(e, ClosedContinuousInterval)
    assert np.isclose(e.length, 0.2)

    e = a + b

    assert d == e

    assert a != b
