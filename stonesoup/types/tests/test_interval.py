import operator
from typing import List

import pytest

from ...types.interval import Interval, Intervals

a = Interval(0, 1)
b = Interval(2, 3)
c = Interval(0.5, 2)
d = Interval(-1, 0)

A = Intervals([a, b])
B = Intervals([c])
C = Intervals([Interval(1, 2)])
D = Intervals([Interval(4, 5), Interval(6, 7)])

attr = operator.attrgetter('left')


def test_interval_init():
    with pytest.raises(ValueError, match="Must have left < right"):
        Interval(0, 0)


def test_interval_len():
    assert a.length == 1
    for i in range(1, 10):
        assert Interval(0, i).length == i


def test_interval_contains():
    assert 0.5 in a
    assert 2 not in a
    assert Interval(0.25, 0.75) in a
    assert Interval(2, 3) not in a
    assert Interval(0.5, 1.5) not in a
    assert 'a string' not in a


def test_interval_str():
    assert str(a) == '[{left}, {right}]'.format(left=0, right=1)
    assert a.__repr__() == 'Interval{interval}'.format(interval=str(a))


def test_interval_eq():
    assert a == Interval(0, 1)
    assert a != (0, 1)
    assert a != Interval(2, 3)


def test_interval_and():
    with pytest.raises(ValueError, match="Can only intersect with Interval types"):
        a & 'a string'  # Can't intersect with non-interval
    assert a & a == a
    assert a & b is None and b & a is None  # Don't intersect
    assert a & c == c & a == Interval(0.5, 1)  # Intersect
    assert a & d is None and d & a is None  # Meet but dont intersect


def test_interval_or():
    with pytest.raises(ValueError, match="Can only union with Interval types"):
        a | 'a string'  # Can't union with non-interval
    assert a | a == [a]
    assert sorted(a | b, key=attr) == sorted(b | a, key=attr) == sorted([a, b], key=attr)
    assert a | c == c | a == [Interval(0, 2)]
    assert a | d == d | a == [Interval(-1, 1)]


def test_interval_sub():
    with pytest.raises(ValueError, match="Can only subtract Interval types from Interval types"):
        a - 'a string'
    assert a - None == [a]
    assert a - a == [None]
    assert a - b == [a] and b - a == [b]
    assert a - Interval(-1, 2) == [None]
    assert a - Interval(-0.5, 0.5) == [Interval(0.5, 1)]
    assert a - c == [Interval(0, 0.5)]
    assert sorted(a - Interval(0.25, 0.75), key=attr) == \
           sorted([Interval(0, 0.25), Interval(0.75, 1)], key=attr)


def test_interval_xor():
    with pytest.raises(ValueError, match="Can only subtract Interval types from Interval types"):
        a ^ 'a string'
    assert a ^ a == [None]
    assert sorted(a ^ b, key=attr) == sorted(b ^ a, key=attr) == sorted([a, b], key=attr)
    assert sorted(a ^ Interval(0.75, 2), key=attr) == \
           sorted([Interval(0, 0.75), Interval(1, 2)], key=attr)
    assert sorted(a ^ Interval(-1, 1), key=attr) == sorted([d], key=attr)


def test_interval_ineq():
    with pytest.raises(ValueError, match="Can only compare Interval types to Interval types"):
        a <= 'a string'
    assert a <= a
    assert a <= Interval(-0.5, 1.5)
    assert not a <= b

    with pytest.raises(ValueError, match="Can only compare Interval types to Interval types"):
        a < 'a string'
    assert a < Interval(-0.5, 1.5)
    assert not a < a
    assert not a < b

    with pytest.raises(ValueError, match="Can only compare Interval types to Interval types"):
        a >= 'a string'
    assert a >= a
    assert a >= Interval(0.25, 0.75)
    assert not a >= b

    with pytest.raises(ValueError, match="Can only compare Interval types to Interval types"):
        a > 'a string'
    assert a > Interval(0.25, 0.75)
    assert not a > a
    assert not a > b


def test_interval_disjoint():
    with pytest.raises(ValueError, match="Interval types can only overlap with Interval types"):
        a.isdisjoint('a string')
    assert a.isdisjoint(b)  # No overlap
    assert not a.isdisjoint(c)  # Overlap
    assert not a.isdisjoint(d)  # Meet and no overlap


def test_intervals_init():
    temp = Intervals()
    assert isinstance(temp.intervals, List)
    assert len(temp.intervals) == 0  # Stores an empty list of intervals

    temp = Intervals(a)
    assert isinstance(temp.intervals, List)
    assert len(temp.intervals) == 1
    assert temp.intervals[0] == a

    with pytest.raises(ValueError, match="Must contain Interval types"):
        Intervals('a string')

    temp = Intervals((0, 1))
    assert isinstance(temp.intervals, List)
    assert len(temp.intervals) == 1
    assert temp.intervals[0] == a  # Converts tuple of length 2 in to a list of one Interval type

    assert isinstance(A.intervals, List)
    assert len(A.intervals) == 2
    assert A.intervals == [a, b]  # Converts lists of tuples to lists of Interval types

    with pytest.raises(ValueError,
                       match="Individual intervals must be an Interval or Sequence type"):
        Intervals([(0, 1), 'a string'])

    temp = Intervals([a, b, c])
    assert isinstance(temp.intervals, List)
    assert temp.intervals == [Interval(0, 3)]  # intervals merge on instantiation


def test_intervals_hash():
    assert hash(Intervals([])) == hash(tuple(list()))
    assert hash(A) == hash(tuple(A.intervals))
    assert hash(B) == hash(tuple(B.intervals))
    assert hash(C) == hash(tuple(C.intervals))
    assert hash(D) == hash(tuple(D.intervals))


def test_intervals_overlap():
    attr = operator.attrgetter('left')

    assert sorted(Intervals.overlap([a, c]), key=attr) == sorted([a, c], key=attr)
    assert Intervals.overlap([a, b]) is None


def test_intervals_disjoint():
    with pytest.raises(ValueError, match="Can only compare Intervals to Intervals"):
        A.isdisjoint('a string')
    assert not A.isdisjoint(B)  # Overlap
    assert not A.isdisjoint(C)  # Meet with no overlap
    assert A.isdisjoint(D)


def test_intervals_merge():
    assert Intervals.get_merged_intervals([a, c]) == [Interval(0, 2)]
    assert sorted(Intervals.get_merged_intervals([a, b]), key=attr) == sorted([a, b], key=attr)


def test_intervals_contains():
    assert 'a string' not in A
    assert 1 in A
    assert 0.5 in A
    assert 1.5 not in A
    assert a in A
    assert b in A
    assert c not in A
    assert A in A
    assert Intervals([a]) in A
    assert Intervals([Interval(0.25, 0.75)]) in A


def test_intervals_str():
    assert str(A) == str([[interval.left, interval.right] for interval in A])
    assert A.__repr__() == 'Intervals{intervals}'.format(intervals=str(A))


def test_intervals_len():
    assert Intervals([]).length == 0
    assert A.length == 2
    assert Intervals([Interval(0.5, 0.75), Interval(0.1, 0.2)]).length == 0.35


def test_intervals_iter():
    intervals = iter([a, b])
    A_iter = iter(A)
    for A_int, interval in zip(A_iter, intervals):
        assert A_int == interval

    intervals = iter([b, a])
    A_iter = reversed(A)
    for A_int, interval in zip(A_iter, intervals):
        assert A_int == interval


def test_intervals_eq():
    assert A != B
    assert A != 'a string'
    assert A == A
    assert A == Intervals([a, b])
    assert B == c


def test_intervals_and():
    with pytest.raises(ValueError, match="Can only intersect with Intervals types"):
        A & 'a string'
    assert A & A == A
    assert A & D == D & A == Intervals([])
    assert A & C == C & A == Intervals([])
    assert A & B == B & A == A & c == Intervals([Interval(0.5, 1)])


def test_intervals_or():
    with pytest.raises(ValueError, match="Can only union with Intervals types"):
        A | 'a string'
    assert A | A == A
    assert A | B == B | A == A | c == Intervals([Interval(0, 3)])
    assert A | D == D | A == Intervals(A.intervals + D.intervals)


def test_intervals_sub():
    with pytest.raises(ValueError, match="Can only subtract Intervals from Intervals"):
        A - 'a string'
    assert A - None == A  # noqa: E711
    assert A - A == Intervals([])
    assert A - a == Intervals([b])
    assert A - C == A
    assert A - Intervals([Interval(0.25, 0.75)]) == \
           Intervals([Interval(0, 0.25), Interval(0.75, 1), b])
    assert A - Intervals([Interval(-1, 4)]) == Intervals([])


def test_intervals_xor():
    with pytest.raises(ValueError, match="Can only compare Intervals from Intervals"):
        A ^ 'a string'
    assert A ^ A == Intervals([])
    assert A ^ B == B ^ A == A ^ c == Intervals([Interval(0, 0.5), Interval(1, 3)])
    assert A ^ D == D ^ A == Intervals(A.intervals + D.intervals)


def test_intervals_ineq():
    with pytest.raises(ValueError, match="Can only compare Intervals to Intervals"):
        A <= 'a string'
    assert A <= A
    assert not A <= B
    assert A <= Intervals(Interval(0, 3))
    assert not A <= a
    assert A <= Interval(0, 3)
    assert A <= Interval(-1, 4)

    with pytest.raises(ValueError, match="Can only compare Intervals to Intervals"):
        A < 'a string'
    assert not A < A
    assert not A < B
    assert A < Intervals(Interval(-1, 4))
    assert not A < a
    assert not A < Interval(0, 3)
    assert A < Interval(-1, 4)

    with pytest.raises(ValueError, match="Can only compare Intervals to Intervals"):
        A >= 'a string'
    assert A >= A
    assert not A >= B
    assert A >= Intervals([Interval(0.25, 0.75), Interval(2.25, 2.75)])
    assert A >= a
    assert not A >= Interval(0, 3)
    assert A >= Interval(0.25, 0.75)

    with pytest.raises(ValueError, match="Can only compare Intervals to Intervals"):
        A > 'a string'
    assert not A > A
    assert not A > B
    assert A > Intervals([Interval(0.25, 0.75), Interval(2.25, 2.75)])
    assert not A > a
    assert not A > Interval(0, 3)
    assert A > Interval(0.25, 0.75)


def test_interval_copy():
    A_copy = A.copy()
    assert A_copy == A
    A_copy.intervals = list()
    assert A_copy.intervals == list()
    assert A_copy.length == 0
    assert A.intervals == [a, b]
    assert A.length == 2


def test_intervals__len__():
    assert len(Intervals([])) == 0
    assert len(A) == 2
    assert len(B) == 1
    assert len(C) == 1
    assert len(D) == 2
    for i in range(1, 10):
        assert len(Intervals([Interval(j + 0.5, j + 1) for j in range(i)])) == i


def test_intervals_remove():
    temp_A = A.copy()
    with pytest.raises(ValueError, match="Intervals only contain Interval types"):
        temp_A.remove('a string')
    with pytest.raises(ValueError, match="Interval not in list"):
        temp_A.remove(c)
    temp_A.remove(b)
    assert temp_A == Intervals([a])


def test_intervals_discard():
    temp_A = A.copy()
    temp_A.discard('a string')
    assert temp_A == Intervals([a, b])
    temp_A.discard(b)
    assert temp_A == Intervals([a])


def test_intervals_pop():
    with pytest.raises(KeyError, match="Contains no intervals"):
        Intervals([]).pop()
    A_temp = A.copy()
    interval = A_temp.pop()
    assert interval in [a, b]
    assert len(A_temp.intervals) == 1
    temp = [a, b]
    temp.remove(interval)
    assert A_temp.intervals == temp


def test_intervals_clear():
    A_temp = A.copy()
    A_temp.clear()
    assert A_temp == Intervals([])
