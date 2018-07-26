# -*- coding: utf-8 -*-
import sys

import pytest

from ..base import Property


def test_properties(base):
    assert 'property_a' in base.properties
    assert base(1, "2").property_a == 1
    assert base(1, "2").property_c == 123


def test_subclass(base):
    class _TestSubclass(base):
        pass
    assert _TestSubclass in base.subclasses


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


@pytest.skip(sys.version_info < (3, 6))
def test_non_base_property():
    with pytest.raises(RuntimeError):
        class _TestNonBase:
            property_a = Property(int)
