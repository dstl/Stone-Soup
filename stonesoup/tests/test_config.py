# -*- coding: utf-8 -*-
import pytest

from ..config import YAMLConfigurationFile
from ..base import Property


@pytest.fixture()
def conf_file():
    return YAMLConfigurationFile()


def test_declarative(base, conf_file):
    instance = base(2, "20")
    instance.new_property = True

    conf_str = conf_file.dumps(instance)

    assert 'property_c' not in conf_str  # Default, no need to store in config

    new_instance = conf_file.load(conf_str)
    assert isinstance(new_instance, base)
    assert new_instance.property_a == instance.property_a
    assert new_instance.property_b == instance.property_b
    assert new_instance.property_c == instance.property_c
    with pytest.raises(AttributeError):
        new_instance.new_property


def test_nested_declarative(base, conf_file):
    nested_instance = base(1, "nested")
    instance = base(2, "primary", nested_instance)

    conf_str = conf_file.dumps(instance)

    new_instance = conf_file.load(conf_str)
    assert isinstance(new_instance, base)
    assert isinstance(new_instance.property_c, base)
    assert new_instance.property_b == "primary"
    assert new_instance.property_c.property_b == "nested"


def test_duplicate_tag_warning(base, conf_file):
    class _TestDuplicateBase(base):
        pass

    class _TestDuplicateBase(base):  # noqa:F801
        pass

    with pytest.warns(UserWarning):
        test_declarative(_TestDuplicateBase, conf_file)


def test_numpy(base, conf_file):
    import numpy as np

    class _TestNumpy(base):
        property_d = Property(np.ndarray, optional=True)

    instance = _TestNumpy(1, "two",
                          property_d=np.array([[1, 2], [3, 4], [5, 6]]))

    conf_str = conf_file.dumps(instance)

    new_instance = conf_file.load(conf_str)
    assert isinstance(new_instance.property_d, np.ndarray)
    assert np.allclose(instance.property_d, new_instance.property_d)
