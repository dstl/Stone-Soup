from typing import List, Any

import pytest

from ..base import Property, Base


def test_properties(base):
    assert 'property_a' in base.properties
    assert base(1, "2").property_a == 1
    assert base(1, "2").property_c == 123


def test_subclass(base):
    class _TestSubclass(base):
        pass
    assert _TestSubclass in base.subclasses


def test_subclass_remove_property(base):
    class _TestSubclassRemoveProperty(base):
        property_a = 2
    assert _TestSubclassRemoveProperty("2").property_a == 2


def test_sub_subclass_remove_property(base):
    class _TestSubclassRemoveProperty(base):
        property_a = 2
    assert _TestSubclassRemoveProperty("2").property_a == 2

    class _TestSubSubclassRemoveProperty(_TestSubclassRemoveProperty):
        pass
    assert _TestSubSubclassRemoveProperty("2").property_a == 2


def test_init_unordered(base):
    with pytest.raises(TypeError):
        class _TestUnordered(base):
            def __init__(self, property_b, *args, **kwrags):
                pass

    with pytest.raises(TypeError):
        class _TestUnordered(base):  # noqa: F811
            def __init__(self, property_b, property_a, *args, **kwrags):
                pass


def test_init_missing(base):
    with pytest.raises(TypeError):
        class _TestMissing(base):
            def __init__(self, property_a, property_b):
                pass

    with pytest.raises(TypeError):
        class _TestMissing(base):  # noqa: F811
            def __init__(self, property_a):
                pass

    with pytest.raises(TypeError):
        class _TestMissing(base):  # noqa: F811
            def __init__(self):
                pass


def test_init_new(base):
    with pytest.raises(TypeError):
        class _TestNew(base):
            def __init__(self, property_d, *args, **kwargs):
                pass

    with pytest.raises(TypeError):
        class _TestNew(base):  # noqa: F811
            def __init__(self, property_d="default", *args, **kwargs):
                pass

    with pytest.raises(TypeError):
        class _TestNew(base):  # noqa: F811
            def __init__(self, property_a, property_b, property_c, property_d):
                pass

    with pytest.raises(TypeError):
        class _TestNew(base):  # noqa: F811
            def __init__(self, *args, property_d, **kwargs):
                pass

    class _TestNew(base):  # noqa: F811
        def __init__(self, *args, property_d="default", **kwargs):
            pass
    assert not hasattr(_TestNew(1, "2", property_d="10"), 'property_d')


def test_non_base_property():
    with pytest.raises(RuntimeError):
        class _TestNonBase:
            property_a = Property(int)


def test_readonly_with_setter():

    class TestReadonly(Base):
        readonly_property_default: float = Property(default=0)
        readonly_property_no_default: float = Property()

        @readonly_property_default.setter
        def readonly_property_default(self, value):
            # assign on first write only
            if not hasattr(self, '_property_readonly_property_default'):
                self._property_readonly_property_default = value
            else:
                # if the value has already been created, the raise an error as
                # it should be read only.
                raise AttributeError

        @readonly_property_no_default.setter
        def readonly_property_no_default(self, value):
            # assign on first write only
            if not hasattr(self, '_property_readonly_property_no_default'):
                self._property_readonly_property_no_default = value
            else:
                # if the value has already been created, the raise an error as
                # it should be read only.
                raise AttributeError

    test_object = TestReadonly(readonly_property_default=10,
                               readonly_property_no_default=20)

    # first test read
    assert test_object.readonly_property_default == 10
    assert test_object.readonly_property_no_default == 20

    # then test that write raises an exception
    with pytest.raises(AttributeError):
        test_object.readonly_property_default = 20
    with pytest.raises(AttributeError):
        test_object.readonly_property_no_default = 10

    # now test when the default is used
    test_object_default = TestReadonly(readonly_property_no_default=20)
    assert test_object_default.readonly_property_default == 0
    with pytest.raises(AttributeError):
        test_object_default.readonly_property_default = 20


def test_basic_setter():

    class TestSetter(Base):
        times_two: float = Property(default=5)

        @times_two.setter
        def times_two(self, value):
            self._property_times_two = value * 2

    test_object = TestSetter(times_two=10)
    assert test_object.times_two == 20
    test_object.times_two = 100
    assert test_object.times_two == 200

    # now test when the default is used
    test_object_default = TestSetter()
    assert test_object_default.times_two == 10


def test_basic_getter():
    # I cannot see a use case where a Stone Soup `Property` with a getter would
    # be appropriate. This would be best done with a Python `property` in all
    # use cases I can see, but is tested here for completeness
    class TestGetter(Base):
        times_two: float = Property(default=None)
        base_value: float = Property(default=5)

        @times_two.getter
        def times_two(self):
            return self.base_value * 2

        # Force the saved value to always be None. Again, don't do this: use a
        # python property
        @times_two.setter
        def times_two(self, value):
            if value is not None:
                raise ValueError
            self._property_times_two = value

    test_object = TestGetter(base_value=10)
    assert test_object.times_two == 20
    test_object.base_value = 100
    assert test_object.times_two == 200

    # now test when the default is used
    test_object_default = TestGetter()
    assert test_object_default.times_two == 10


def test_readonly():

    class TestReadonly(Base):
        readonly_property_default: float = Property(default=0, readonly=True)
        readonly_property_no_default: float = Property(readonly=True)

    test_object = TestReadonly(readonly_property_default=10,
                               readonly_property_no_default=20)

    # first test read
    assert test_object.readonly_property_default == 10
    assert test_object.readonly_property_no_default == 20

    # then test that write raises an exception
    with pytest.raises(AttributeError):
        test_object.readonly_property_default = 20
    with pytest.raises(AttributeError):
        test_object.readonly_property_no_default = 10

    # now test when the default is used
    test_object_default = TestReadonly(readonly_property_no_default=20)
    assert test_object_default.readonly_property_default == 0
    with pytest.raises(AttributeError):
        test_object_default.readonly_property_default = 20


def test_readonly_subclass():

    class TestParent(Base):
        readonly_property_default: float = Property(default=0, readonly=True)
        readonly_property_no_default: float = Property(readonly=True)

    class TestReadonly(TestParent):
        pass

    test_object = TestReadonly(readonly_property_default=10,
                               readonly_property_no_default=20)

    # first test read
    assert test_object.readonly_property_default == 10
    assert test_object.readonly_property_no_default == 20

    # then test that write raises an exception
    with pytest.raises(AttributeError):
        test_object.readonly_property_default = 20
    with pytest.raises(AttributeError):
        test_object.readonly_property_no_default = 10

    # now test when the default is used
    test_object_default = TestReadonly(readonly_property_no_default=20)
    assert test_object_default.readonly_property_default == 0
    with pytest.raises(AttributeError):
        test_object_default.readonly_property_default = 20


def test_readonly_with_getter():

    class TestReadonly(Base):
        readonly_property_default: float = Property(default=0, readonly=True)
        readonly_property_no_default: float = Property(readonly=True)

        @readonly_property_default.getter
        def readonly_property_default(self):
            return self._property_readonly_property_default

    test_object = TestReadonly(readonly_property_default=10,
                               readonly_property_no_default=20)

    # first test read
    assert test_object.readonly_property_default == 10
    assert test_object.readonly_property_no_default == 20

    # then test that write raises an exception
    with pytest.raises(AttributeError):
        test_object.readonly_property_default = 20
    with pytest.raises(AttributeError):
        test_object.readonly_property_no_default = 10

    # now test when the default is used
    test_object_default = TestReadonly(readonly_property_no_default=20)
    assert test_object_default.readonly_property_default == 0
    with pytest.raises(AttributeError):
        test_object_default.readonly_property_default = 20


def test_type_hint_checking():
    """ Check that errors are raised for some common type hint errors """
    # no error
    class TestClass(Base):
        i: int = Property(doc='Test')
    _ = TestClass(i=1)

    # no error
    class TestClass(Base):
        i = Property(int, doc='Test')
    _ = TestClass(i=1)

    # specify both as a type hint AND and argument
    with pytest.raises(ValueError):
        class TestClass(Base):
            i: float = Property(int, doc='Test')
        obj = TestClass(i=1)
        assert obj._properties['i'].cls is int

    with pytest.raises(ValueError):
        # Type is not specified
        class TestClass(Base):
            i = Property(doc='Test')
        _ = TestClass(i=1)

    # error for [int]
    with pytest.raises(ValueError):
        class TestClass(Base):
            i: [int] = Property(doc='Test')
        _ = TestClass(i=1)

    with pytest.raises(ValueError):
        class TestClass(Base):
            i = Property([int], doc='Test')
        _ = TestClass(i=1)

    # No error for List[int]
    class TestClass(Base):
        i: List[int] = Property(doc='Test')
    _ = TestClass(i=1)

    class TestClass(Base):
        i = Property(List[int], doc='Test')
    _ = TestClass(i=1)

    with pytest.raises(ValueError):
        class TestClass(Base):
            i: 'string' = Property(doc='Test')  # noqa: F821
        _ = TestClass(i=1)

    with pytest.raises(ValueError):
        class TestClass(Base):
            i = Property('string', doc='Test')
        _ = TestClass(i=1)

    # errors for any
    with pytest.raises(ValueError):
        class TestClass(Base):
            i: any = Property(doc='Test')
        _ = TestClass(i=1)

    with pytest.raises(ValueError):
        class TestClass(Base):
            i = Property(any, doc='Test')
        _ = TestClass(i=1)

    # no error for typing.Any
    class TestClass(Base):
        i: Any = Property(doc='Test')
    _ = TestClass(i=1)

    # no error
    class TestClass(Base):
        i = Property(Any, doc='Test')
    _ = TestClass(i=1)
