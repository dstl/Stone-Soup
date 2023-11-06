import datetime

import numpy as np
import pytest
from ruamel.yaml.constructor import ConstructorError

from .conftest import _TestBase
from ..sensor.sensor import Sensor
from ..serialise import YAML
from ..base import Property
from ..types.array import Matrix, StateVector, CovarianceMatrix
from ..types.angle import Angle, Bearing, Elevation, Longitude, Latitude


@pytest.fixture(params=['rt', 'safe'])
def serialised_file(request):
    return YAML(typ=request.param)


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


class _TestDuplicateBase(_TestBase):
    pass


first_class = _TestDuplicateBase


class _TestDuplicateBase(_TestBase):  # noqa:F801
    pass


second_class = _TestDuplicateBase


def test_duplicate_tag_warning(serialised_file):
    if 'safe' in serialised_file.typ:
        pytest.xfail("With 'safe' constructor, warning isn't raised")

    instance = first_class(2, "20")
    instance.new_property = True

    serialised_str = serialised_file.dumps(instance)

    with pytest.warns(UserWarning):
        new_instance = serialised_file.load(serialised_str)

    assert isinstance(new_instance, (first_class, second_class))


@pytest.mark.parametrize(
    'instance',
    [Angle(0.1), Bearing(0.2), Elevation(0.3), Longitude(0.4), Latitude(0.5)],
    ids=('Angle', 'Bearing', 'Elevation', 'Longitude', 'Latitude'))
def test_angle(serialised_file, instance):
    serialised_str = serialised_file.dumps(instance)

    new_instance = serialised_file.load(serialised_str)
    assert isinstance(new_instance, type(instance))
    assert instance == new_instance


def test_probability(serialised_file):
    from ..types.numeric import Probability

    instance = Probability(1E-100)

    serialised_str = serialised_file.dumps(instance)
    assert 'exp' not in serialised_str

    new_instance = serialised_file.load(serialised_str)
    assert isinstance(new_instance, Probability)
    assert instance == new_instance

    instance = instance**4  # Very small number, so need log representation
    serialised_str = serialised_file.dumps(instance)
    assert 'exp' in serialised_str
    assert str(instance.log_value) in serialised_str

    new_instance = serialised_file.load(serialised_str)
    assert isinstance(new_instance, Probability)
    assert instance == new_instance


class _TestNumpy(_TestBase):
    property_d: np.ndarray = Property()


def test_numpy(serialised_file):
    instance = _TestNumpy(1, "two",
                          property_d=np.array([[1, 2], [3, 4], [5, 6]]))

    serialised_str = serialised_file.dumps(instance)

    new_instance = serialised_file.load(serialised_str)
    assert isinstance(new_instance.property_d, np.ndarray)
    assert np.allclose(instance.property_d, new_instance.property_d)


@pytest.mark.parametrize(
    'instance',
    [Matrix([[1, 2, 4], [4, 5, 6]]),
     StateVector([[1], [2], [3], [4]]),
     CovarianceMatrix([[1, 0], [0, 2]])],
    ids=('Matrix', 'StateVector', 'CovarianceMatrix')
)
def test_arrays(serialised_file, instance):
    serialised_str = serialised_file.dumps(instance)

    new_instance = serialised_file.load(serialised_str)
    assert isinstance(new_instance, type(instance))
    assert np.allclose(instance, new_instance)


@pytest.mark.parametrize(
    'values',
    [[np.int_(10), np.int16(20), np.int64(-30)],
     [np.float_(10.0), np.float64(20.1), np.float32(-0.5), np.longfloat(10.5)]],
    ids=['int', 'float'])
def test_numpy_dtypes(serialised_file, values):
    serialised_str = serialised_file.dumps(values)
    new_values = serialised_file.load(serialised_str)
    assert new_values == values


class _TestDatetime(_TestBase):
    property_d: datetime.datetime = Property()


def test_datetime(serialised_file):
    instance = _TestDatetime(1, "two",
                             property_d=datetime.timedelta(seconds=500))

    serialised_str = serialised_file.dumps(instance)

    new_instance = serialised_file.load(serialised_str)
    assert isinstance(new_instance.property_d, datetime.timedelta)
    assert instance.property_d == new_instance.property_d


def test_deque(base, serialised_file):
    from collections import deque

    max_len = 5
    instance = deque([3, 4, 5, 6, 7, 8, 9], max_len)

    serialised_str = serialised_file.dumps(instance)

    new_instance = serialised_file.load(serialised_str)
    assert new_instance == instance


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
        test1: &id001 !stonesoup.tests.conftest._TestBase
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


class TestSensor(Sensor):
    @property
    def measurement_model(self):
        raise NotImplementedError

    def measure(self):
        pass


def test_sensor_serialisation(serialised_file):
    sensor = TestSensor()
    assert sensor.position is None
    assert sensor.orientation is None
    serialised_str = serialised_file.dumps(sensor)
    sensor = serialised_file.load(serialised_str)
    assert sensor.position is None
    assert sensor.orientation is None

    pos = StateVector([0, 1, 2])
    orientation = StateVector([0, np.pi/2, np.pi/4])
    sensor = TestSensor(position=pos, orientation=orientation)
    assert np.allclose(sensor.position, pos)
    assert np.allclose(sensor.orientation, orientation)
    serialised_str = serialised_file.dumps(sensor)
    sensor = serialised_file.load(serialised_str)
    assert np.allclose(sensor.position, pos)
    assert np.allclose(sensor.orientation, orientation)


def test_dump(tmpdir, serialised_file):
    data = [1, 2, 3]
    with open(tmpdir.join('dump_file.yml'), 'w') as yaml_file:
        serialised_file.dump(data, yaml_file)

    with open(tmpdir.join('dump_file.yml')) as yaml_file:
        read_data = serialised_file.load(yaml_file)

    assert read_data == data


def test_dump_all(tmpdir, serialised_file):
    documents = [[i, i + 1, i + 2] for i in range(5)]
    with open(tmpdir.join('dump_file.yml'), 'w') as yaml_file:
        serialised_file.dump_all(documents, yaml_file)

    with open(tmpdir.join('dump_file.yml')) as yaml_file:
        read_documents = serialised_file.load_all(yaml_file)

        for read_document, document in zip(read_documents, documents):
            assert read_document == document
