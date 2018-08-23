# -*- coding: utf-8 -*-
import pytest
from ruamel.yaml.constructor import ConstructorError

from ..serialise import YAML
from ..base import Property


@pytest.fixture()
def serialised_file():
    return YAML()


def test_declarative(base, serialised_file):
    instance = base(2, "20")
    instance.new_property = True

    serialised_str = serialised_file.dumps(instance)

    assert 'property_c' not in serialised_str  # Default, no need to store

    new_instance = serialised_file.load(serialised_str)
    assert isinstance(new_instance, base)
    assert new_instance.property_a == instance.property_a
    assert new_instance.property_b == instance.property_b
    assert new_instance.property_c == instance.property_c
    with pytest.raises(AttributeError):
        new_instance.new_property


def test_nested_declarative(base, serialised_file):
    nested_instance = base(1, "nested")
    instance = base(2, "primary", nested_instance)

    serialised_str = serialised_file.dumps(instance)

    new_instance = serialised_file.load(serialised_str)
    assert isinstance(new_instance, base)
    assert isinstance(new_instance.property_c, base)
    assert new_instance.property_b == "primary"
    assert new_instance.property_c.property_b == "nested"


def test_duplicate_tag_warning(base, serialised_file):
    class _TestDuplicateBase(base):
        pass

    first_class = _TestDuplicateBase

    class _TestDuplicateBase(base):  # noqa:F801
        pass

    second_class = _TestDuplicateBase

    instance = first_class(2, "20")
    instance.new_property = True

    serialised_str = serialised_file.dumps(instance)

    with pytest.warns(UserWarning):
        new_instance = serialised_file.load(serialised_str)

    assert isinstance(new_instance, (first_class, second_class))


def test_numpy(base, serialised_file):
    import numpy as np

    class _TestNumpy(base):
        property_d = Property(np.ndarray)

    instance = _TestNumpy(1, "two",
                          property_d=np.array([[1, 2], [3, 4], [5, 6]]))

    serialised_str = serialised_file.dumps(instance)

    new_instance = serialised_file.load(serialised_str)
    assert isinstance(new_instance.property_d, np.ndarray)
    assert np.allclose(instance.property_d, new_instance.property_d)


def test_datetime(base, serialised_file):
    import datetime

    class _TestNumpy(base):
        property_d = Property(datetime.timedelta)

    instance = _TestNumpy(1, "two",
                          property_d=datetime.timedelta(seconds=500))

    serialised_str = serialised_file.dumps(instance)

    new_instance = serialised_file.load(serialised_str)
    assert isinstance(new_instance.property_d, datetime.timedelta)
    assert instance.property_d == new_instance.property_d


def test_path(serialised_file):
    import pathlib
    import tempfile

    with tempfile.NamedTemporaryFile() as file:
        path = pathlib.Path(file.name)
        serialised_str = serialised_file.dumps(path)
        assert file.name in serialised_str

        new_path = serialised_file.load(serialised_str)
        assert new_path == path


def test_references(base, serialised_file):
    serialised_str = """
        property_b: &prop_b '20'
        test1: &id001 !stonesoup.tests.conftest.base.%3Clocals%3E._TestBase
            - property_a: 2
            - property_b: *prop_b
        test2: *id001
        """

    conf = serialised_file.load(serialised_str)
    new_instance = conf['test2']
    assert new_instance is conf['test1']
    assert isinstance(new_instance, base)
    assert new_instance.property_a == 2
    assert new_instance.property_b == "20"
    assert new_instance.property_c == base.property_c.default


def test_anchor(base, serialised_file):
    instance = base(2, "20")

    serialised_str = serialised_file.dumps(
        [instance, instance,  {"key": instance}])
    assert '&id001' in serialised_str  # Anchor should be created
    assert '*id001' in serialised_str  # Reference should be created

    new_instances = serialised_file.load(serialised_str)
    assert new_instances[0] is new_instances[1]
    assert new_instances[0] is new_instances[2]['key']


def test_bad_tag(serialised_file):
    # Invalid module
    serialised_str = """
        test1: &id001 !stonesoup.tests.this.does.not.exist
            - property_a: 2
            - property_b: "10"
        """

    with pytest.raises(ConstructorError, match="unable to import component"):
        serialised_file.load(serialised_str)

    # Invalid class in valid module
    serialised_str = """
        test1: &id001 !stonesoup.tests.conftest.nope
            - property_a: 2
            - property_b: "10"
        """

    with pytest.raises(ConstructorError, match="unable to import component"):
        serialised_file.load(serialised_str)


def test_missing_property(base, serialised_file):
    instance = base(2, "20")
    instance.new_property = True

    # Pass over `property_b` line.
    serialised_str = "\n".join(
        line
        for line in serialised_file.dumps(instance).split("\n")
        if 'property_b' not in line)

    assert 'property_b' not in serialised_str  # Default, no need to store

    with pytest.raises(ConstructorError, match="missing a required argument"):
        serialised_file.load(serialised_str)
