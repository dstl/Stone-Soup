import pytest

from stonesoup.base import ImmutableMixIn
from stonesoup.base import Property, Base, Freezable


class TestClass(Base, ImmutableMixIn):
    a: int = Property()
    b: float = Property()


class TestSubClass(TestClass):
    c: int = Property()
    d: str = Property()


def test_simple_case():
    obj = TestClass(a=1, b=1.2)
    with pytest.raises(AttributeError, match='a is readonly'):
        obj.a = 2
    with pytest.raises(AttributeError, match='b is readonly'):
        obj.b = None


def test_subclass():
    obj = TestSubClass(a=1, b=1.2, c=0, d='str')
    with pytest.raises(AttributeError, match='a is readonly'):
        obj.a = 2
    with pytest.raises(AttributeError, match='b is readonly'):
        obj.b = None
    with pytest.raises(AttributeError, match='c is readonly'):
        obj.c = 2
    with pytest.raises(AttributeError, match='d is readonly'):
        obj.d = None


def test_copy():
    obj = TestClass(a=1, b=1.2)
    obj2 = obj.copy_with_updates(b=2.1)
    assert obj.a == 1
    assert obj.b == 1.2
    assert obj2.a == 1
    assert obj2.b == 2.1
    assert obj != obj2


def test_copy_subclass():
    obj = TestSubClass(a=1, b=1.2, c=0, d='str')
    obj2 = obj.copy_with_updates(b=2.1)
    assert obj.a == 1
    assert obj.b == 1.2
    assert obj.c == 0
    assert obj.d == 'str'
    assert obj2.a == 1
    assert obj2.b == 2.1
    assert obj.c == 0
    assert obj.d == 'str'
    assert obj != obj2


def test_equality():
    obj = TestClass(a=1, b=0)
    obj2 = TestClass(a=1, b=0)
    assert obj._is_hashable
    assert obj2._is_hashable
    assert obj is not obj2
    assert obj == obj2

    # If we use a non-hashable value, then equality won't work by value
    obj = TestClass(a=1, b=[1])
    obj2 = TestClass(a=1, b=[1])
    assert not obj._is_hashable
    assert not obj2._is_hashable
    assert obj is not obj2
    assert obj != obj2
    # but it should work by identity
    obj2 = obj
    assert obj == obj2


def test_equality_subclass():
    obj = TestSubClass(a=1, b=0, c=0, d='str')
    obj2 = TestSubClass(a=1, b=0, c=0, d='str')
    obj3 = TestClass(a=1, b=0)
    assert obj is not obj2
    assert obj == obj2
    assert obj2 != obj3

    # If we use a non-hashable value, then equality won't work by value
    obj = TestSubClass(a=1, b=[1], c=5, d='test')
    obj2 = TestSubClass(a=1, b=[1], c=5, d='test')
    assert obj is not obj2
    assert obj != obj2
    # but it should work by identity
    obj2 = obj
    assert obj == obj2

    # repeat with a non-inherited property
    obj = TestSubClass(a=1, b=1, c=5, d=['test'])
    obj2 = TestSubClass(a=1, b=1, c=5, d=['test'])
    assert obj is not obj2
    assert obj != obj2
    # but it should work by identity
    obj2 = obj
    assert obj == obj2


def test_freezing():
    @Freezable
    class TestClass(Base):
        var1: int = Property()
        var2: float = Property()

    objs = [TestClass(var1=i, var2=0.5*i) for i in range(4)]

    assert all(isinstance(obj, TestClass) for obj in objs)
    # first check we can set things....
    for obj in objs:
        obj.var1 = 1
        obj.var2 = None
    # then freeze them explicitly:
    objs = [obj.freeze() for obj in objs]

    # noinspection PyUnresolvedReferences
    assert all(obj.__class__.__name__ == 'FrozenTestClass' for obj in objs)
    # Now we should get an error
    for obj in objs:
        with pytest.raises(AttributeError, match='var1 is readonly'):
            obj.var1 = 2
        with pytest.raises(AttributeError, match='var2 is readonly'):
            obj.var2 = 3
