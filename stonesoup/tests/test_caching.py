from stonesoup.base import MutableCacheableProperty, Base, Property


def test_cache_creation():
    """Test that :class:`MutableCacheableProperty` correctly caches a property value and then
    clears it when a property is updated."""
    @MutableCacheableProperty()
    class TestClass:
        def __init__(self):
            self.a = 3
            self.b = 4
            self.calls = 0

        @property
        @MutableCacheableProperty.cached_property()
        def checksum(self):
            self.calls += 1
            return self.a * 1 + self.b * 2

    obj = TestClass()
    assert obj.a == 3
    assert hasattr(obj, '_cached_properties')
    assert hasattr(TestClass, '_cached_properties')
    assert obj._cached_properties == set()
    assert TestClass._cached_properties == set()

    assert obj.checksum == 11
    assert obj.calls == 1
    assert 'checksum' in obj._cached_properties
    assert len(obj._cached_properties) == 1

    # test that another call doesn't change things
    assert obj.checksum == 11
    assert obj.calls == 1
    assert 'checksum' in obj._cached_properties
    assert len(obj._cached_properties) == 1

    # try changing the value
    obj.b = 2
    # check cache is cleared
    assert obj._checksum_cache is None
    # check value recalculates
    assert obj.checksum == 7
    assert obj.calls == 2

    # Another call should use the cached value
    assert obj.checksum == 7
    assert obj.calls == 2


def test_cache_dependency():
    """Test that :class:`MutableCacheableProperty` correctly caches a property value and then
    clears it when a another object on which the property depends is updated."""
    @MutableCacheableProperty()
    class ChildClass:
        def __init__(self):
            self.c = 1
            self.d = 4

    @MutableCacheableProperty()
    class TestClass(Base):
        a: int = Property()
        child: ChildClass = Property()

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.b = 4
            self.calls = 0

        @property
        @MutableCacheableProperty.cached_property(depends_on=[child])
        def checksum(self):
            self.calls += 1
            return self.a * 1 + self.b * 2 + self.child.c * 3 + self.child.d * 4

    obj = TestClass(a=3, child=ChildClass())
    assert obj.a == 3
    child = obj.child
    assert child.d == 4

    assert obj.checksum == 30
    assert obj.calls == 1

    # test that another call doesn't change things
    assert obj.checksum == 30
    assert obj.calls == 1

    assert child._TestClass_checksum_cache is True
    # try changing the value
    child.d = 2
    # check cache is cleared
    assert child._TestClass_checksum_cache is None
    # check value recalculates
    assert obj.checksum == 22
    assert obj.calls == 2

    # Another call should use the cached value
    assert obj.checksum == 22
    assert obj.calls == 2


def test_cache_dependency_python_property():
    """Test that :class:`MutableCacheableProperty` correctly caches a property value and then
    clears it when a another object on which the property depends is updated if the two
    objects are linked as Python properties rather than StoneSoup properties."""
    @MutableCacheableProperty()
    class ChildClass:
        def __init__(self):
            self.c = 1
            self.d = 4

    @MutableCacheableProperty()
    class TestClass(Base):
        a: int = Property()
        child: ChildClass = Property()

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.b = 4
            self.calls = 0

        @property
        def child_prop(self):
            return self.child

        @property
        @MutableCacheableProperty.cached_property(depends_on=[child_prop])
        def checksum(self):
            self.calls += 1
            return self.a * 1 + self.b * 2 + self.child.c * 3 + self.child.d * 4

    obj = TestClass(a=3, child=ChildClass())
    assert obj.a == 3
    child = obj.child
    assert child.d == 4

    assert obj.checksum == 30
    assert obj.calls == 1

    # test that another call doesn't change things
    assert obj.checksum == 30
    assert obj.calls == 1

    assert child._TestClass_checksum_cache is True
    # try changing the value
    child.d = 2
    # check cache is cleared
    assert child._TestClass_checksum_cache is None
    # check value recalculates
    assert obj.checksum == 22
    assert obj.calls == 2

    # Another call should use the cached value
    assert obj.checksum == 22
    assert obj.calls == 2
