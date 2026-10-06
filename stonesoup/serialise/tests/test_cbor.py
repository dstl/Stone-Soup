import datetime
import json
from collections import deque
from pathlib import Path

import numpy as np
import pytest

cbor2 = pytest.importorskip("cbor2")

from .test_yaml import TestSensor, first_class, second_class  # noqa: E402
from .. import CBOR  # noqa: E402
from ..cbor import encode_object  # noqa: E402
from ..yaml import get_class  # noqa: E402
from ...base import Property  # noqa: E402
from ...tests.conftest import _TestBase  # noqa: E402
from ...types.angle import Angle, Bearing, Elevation, Longitude, Latitude  # noqa: E402
from ...types.array import Matrix, StateVector, StateVectors, CovarianceMatrix  # noqa: E402
from ...types.numeric import Probability  # noqa: E402
from ...types.state import GaussianState, ParticleState  # noqa: E402
from ...types.track import Track  # noqa: E402


@pytest.fixture(params=[True, False], ids=['sharing', 'no_sharing'])
def serialiser(request):
    return CBOR(value_sharing=request.param, string_referencing=request.param)


def test_declarative(base, serialiser):
    instance = base(2, "20")
    instance.new_property = True

    data = serialiser.dumps(instance)
    assert b'property_c' not in data  # Default, no need to store

    new_instance = serialiser.loads(data)
    assert isinstance(new_instance, base)
    assert new_instance.property_a == instance.property_a
    assert new_instance.property_b == instance.property_b
    assert new_instance.property_c == instance.property_c
    with pytest.raises(AttributeError):
        new_instance.new_property


def test_nested_declarative(base, serialiser):
    nested_instance = base(1, "nested")
    instance = base(2, "primary", nested_instance)

    new_instance = serialiser.loads(serialiser.dumps(instance))
    assert isinstance(new_instance, base)
    assert isinstance(new_instance.property_c, base)
    assert new_instance.property_b == "primary"
    assert new_instance.property_c.property_b == "nested"


def test_duplicate_type_name_warning(serialiser):
    get_class.cache_clear()  # Warning only raised on first lookup

    instance = first_class(2, "20")
    data = serialiser.dumps(instance)

    try:
        with pytest.warns(UserWarning):
            new_instance = serialiser.loads(data)
    finally:
        get_class.cache_clear()  # Ensure other tests also get warning

    assert isinstance(new_instance, (first_class, second_class))


@pytest.mark.parametrize(
    'instance',
    [Angle(0.1), Bearing(0.2), Elevation(0.3), Longitude(0.4), Latitude(0.5)],
    ids=('Angle', 'Bearing', 'Elevation', 'Longitude', 'Latitude'))
def test_angle(serialiser, instance):
    new_instance = serialiser.loads(serialiser.dumps(instance))
    assert isinstance(new_instance, type(instance))
    assert instance == new_instance


@pytest.mark.parametrize('instance', [Probability(1E-100), Probability(1E-100)**4, Probability(0)])
def test_probability(serialiser, instance):
    new_instance = serialiser.loads(serialiser.dumps(instance))
    assert isinstance(new_instance, Probability)
    assert instance == new_instance


class _TestNumpy(_TestBase):
    property_d: np.ndarray = Property()


def test_numpy(serialiser):
    instance = _TestNumpy(1, "two", property_d=np.array([[1, 2], [3, 4], [5, 6]]))

    new_instance = serialiser.loads(serialiser.dumps(instance))
    assert type(new_instance.property_d) is np.ndarray
    assert np.array_equal(instance.property_d, new_instance.property_d)


@pytest.mark.parametrize(
    'dtype', ['<f8', '>f8', '<f4', 'f2', '<i8', '>i4', 'i2', 'i1', 'u1', '<u2', 'u4', 'u8',
              'bool', 'complex128', 'longdouble'])
@pytest.mark.parametrize('shape', [(3, 2), (4, ), (2, 3, 4), (), (0, 3)])
def test_numpy_array_dtypes(serialiser, dtype, shape):
    array = (np.arange(int(np.prod(shape))) % 7).reshape(shape).astype(dtype)
    new_array = serialiser.loads(serialiser.dumps(array))
    assert new_array.shape == array.shape
    if array.size or dtype not in ('bool', 'complex128', 'longdouble'):
        # Typed arrays keep dtype (in native byte order); generic arrays only when not empty
        assert new_array.dtype == array.dtype.newbyteorder('=') or dtype == 'longdouble'
    assert np.array_equal(new_array, array)
    if array.size:
        new_array.flat[0] = 1  # Should be writable


def test_numpy_non_contiguous(serialiser):
    array = np.arange(12.).reshape(3, 4)[:, ::2]
    assert np.array_equal(serialiser.loads(serialiser.dumps(array)), array)
    assert np.array_equal(serialiser.loads(serialiser.dumps(array.T)), array.T)


def test_numpy_object_array(serialiser):
    array = StateVector([Bearing(0.1), Elevation(0.2), 3.])
    new_array = serialiser.loads(serialiser.dumps(array))
    assert isinstance(new_array, StateVector)
    assert isinstance(new_array[0, 0], Bearing)
    assert isinstance(new_array[1, 0], Elevation)
    assert new_array[2, 0] == 3.


@pytest.mark.parametrize(
    'instance',
    [Matrix([[1, 2, 4], [4, 5, 6]]),
     StateVector([[1], [2], [3], [4]]),
     StateVectors([[1., 2.], [3., 4.]]),
     CovarianceMatrix([[1, 0], [0, 2]])],
    ids=('Matrix', 'StateVector', 'StateVectors', 'CovarianceMatrix')
)
def test_arrays(serialiser, instance):
    new_instance = serialiser.loads(serialiser.dumps(instance))
    assert type(new_instance) is type(instance)
    assert np.array_equal(instance, new_instance)


@pytest.mark.parametrize(
    'values',
    [[np.int_(10), np.int16(20), np.int64(-30)],
     [np.float64(20.1), np.float32(-0.5), np.longdouble(10.5)],
     [np.bool_(True), np.bool_(False)]],
    ids=['int', 'float', 'bool'])
def test_numpy_dtypes(serialiser, values):
    assert serialiser.loads(serialiser.dumps(values)) == values


@pytest.mark.parametrize(
    'instance',
    [datetime.datetime(2024, 1, 2, 3, 4, 5, 6),
     datetime.datetime(2024, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc),
     datetime.datetime(2024, 1, 2, 3, 4, 5, tzinfo=datetime.timezone(datetime.timedelta(hours=1))),
     datetime.timedelta(seconds=500),
     datetime.timedelta(days=-3, microseconds=1),
     datetime.timedelta(days=10000, microseconds=999999)],
    ids=['naive', 'utc', 'offset', 'timedelta', 'negative_timedelta', 'large_timedelta'])
def test_datetime(serialiser, instance):
    new_instance = serialiser.loads(serialiser.dumps(instance))
    assert type(new_instance) is type(instance)
    assert new_instance == instance
    if isinstance(instance, datetime.datetime):
        assert new_instance.utcoffset() == instance.utcoffset()


def test_datetime_naive_no_timezone(serialiser):
    # Default: stored without offset and loaded as naive. As RFC 3339 (tag 0) requires an
    # offset, stored with type name (tag 27) instead.
    instance = datetime.datetime(2024, 1, 2, 3, 4, 5, 6)
    data = serialiser.dumps(instance)
    assert b'2024-01-02T03:04:05.000006' in data
    assert b'2024-01-02T03:04:05.000006+' not in data
    raw = cbor2.loads(data)
    assert isinstance(raw, cbor2.CBORTag) and raw.tag == 27
    assert list(raw.value) == ['datetime.datetime', '2024-01-02T03:04:05.000006']
    new_instance = serialiser.loads(data)
    assert new_instance.tzinfo is None
    assert new_instance == instance


@pytest.mark.parametrize(
    'timezone, expected_offset',
    [(datetime.timezone.utc, '+00:00'),
     (datetime.timezone(datetime.timedelta(hours=-5)), '-05:00'),
     ('Europe/London', '+01:00')],  # Summer time
    ids=['utc', 'fixed_offset', 'zoneinfo'])
def test_datetime_naive_with_timezone(timezone, expected_offset):
    if isinstance(timezone, str):
        zoneinfo = pytest.importorskip('zoneinfo')
        try:
            timezone = zoneinfo.ZoneInfo(timezone)
        except zoneinfo.ZoneInfoNotFoundError:
            pytest.skip("Timezone data not available")
    serialiser = CBOR(timezone=timezone)
    instance = datetime.datetime(2024, 7, 2, 3, 4, 5, 6)

    data = serialiser.dumps(instance)
    assert f'2024-07-02T03:04:05.000006{expected_offset}'.encode() in data
    assert isinstance(cbor2.loads(data), datetime.datetime)  # Standard RFC 3339 string (tag 0)

    new_instance = serialiser.loads(data)
    assert new_instance.utcoffset() is not None  # Now timezone aware
    assert new_instance == instance.replace(tzinfo=timezone)
    assert new_instance.replace(tzinfo=None) == instance  # Same wall clock time


def test_datetime_aware_with_timezone():
    # Timezone aware datetimes unaffected by timezone option
    serialiser = CBOR(timezone=datetime.timezone.utc)
    instance = datetime.datetime(
        2024, 1, 2, 3, 4, 5, tzinfo=datetime.timezone(datetime.timedelta(hours=1)))
    new_instance = serialiser.loads(serialiser.dumps(instance))
    assert new_instance == instance
    assert new_instance.utcoffset() == datetime.timedelta(hours=1)


def test_datetime_timezone_nested():
    # Applies to datetimes within components, e.g. state timestamps
    serialiser = CBOR(timezone=datetime.timezone.utc)
    track = Track([GaussianState([1, 2], np.eye(2), timestamp=datetime.datetime(2024, 1, 2))])
    new_track = serialiser.loads(serialiser.dumps(track))
    assert new_track.timestamp == datetime.datetime(2024, 1, 2, tzinfo=datetime.timezone.utc)


def test_bad_timezone():
    with pytest.raises(TypeError, match="timezone must be a datetime.tzinfo"):
        CBOR(timezone="UTC")


def test_deque(serialiser):
    instance = deque([3, 4, 5, 6, 7, 8, 9], 5)
    new_instance = serialiser.loads(serialiser.dumps(instance))
    assert new_instance == instance
    assert new_instance.maxlen == instance.maxlen


def test_path(serialiser):
    path = Path('/some/path/file.txt')
    assert serialiser.loads(serialiser.dumps(path)) == path


def test_sets(serialiser):
    tracks = {Track([GaussianState([1, 2], np.eye(2))]), Track()}
    new_tracks = serialiser.loads(serialiser.dumps({'tracks': tracks}))['tracks']
    assert isinstance(new_tracks, set)
    assert {len(track) for track in new_tracks} == {0, 1}
    for track in new_tracks:
        assert isinstance(track.states, list)  # Not immutable tuple
    assert serialiser.loads(serialiser.dumps(frozenset({1, 2}))) == frozenset({1, 2})


def test_references(base, serialiser):
    instance = base(2, "20")
    array = StateVector([1, 2])
    a_list = [1, 2]

    new_instances = serialiser.loads(serialiser.dumps(
        [instance, instance, {"key": instance}, array, array, a_list, a_list]))
    if serialiser.value_sharing:
        assert new_instances[0] is new_instances[1]
        assert new_instances[0] is new_instances[2]['key']
        assert new_instances[3] is new_instances[4]
        assert new_instances[5] is new_instances[6]
    else:
        assert new_instances[0] is not new_instances[1]
    assert new_instances[1].property_b == "20"


def test_value_sharing_reduces_size(base):
    instance = base(2, "20", base(3, "nested"))
    data = [instance] * 100
    assert len(CBOR().dumps(data)) * 5 < len(CBOR(value_sharing=False).dumps(data))


def test_bad_type_name(serialiser):
    # Invalid module
    data = cbor2.dumps(cbor2.CBORTag(27, ['stonesoup.tests.this.does.not.exist', {}]))
    with pytest.raises(cbor2.CBORDecodeError, match="unable to import component"):
        serialiser.loads(data)

    # Invalid class in valid module
    data = cbor2.dumps(cbor2.CBORTag(27, ['stonesoup.tests.conftest.nope', {}]))
    with pytest.raises(cbor2.CBORDecodeError, match="unable to import component"):
        serialiser.loads(data)

    # Not a Stone Soup component
    data = cbor2.dumps(cbor2.CBORTag(27, ['stonesoup.base.Property', {}]))
    with pytest.raises(cbor2.CBORDecodeError, match="not a component"):
        serialiser.loads(data)

    # Outside of Stone Soup, and not already imported
    data = cbor2.dumps(cbor2.CBORTag(27, ['os.system', {}]))
    with pytest.raises(cbor2.CBORDecodeError, match="unknown type"):
        serialiser.loads(data)


def test_missing_property(base, serialiser):
    data = cbor2.dumps(cbor2.CBORTag(
        27, ['stonesoup.tests.conftest._TestBase', {'property_a': 2}]))
    with pytest.raises(cbor2.CBORDecodeError, match="missing a required argument"):
        serialiser.loads(data)


def test_unsupported_type(serialiser):
    with pytest.raises(cbor2.CBOREncodeError, match="cannot serialise"):
        serialiser.dumps(object())


def test_non_container_top_level(base, serialiser):
    # Nested repeated strings, with component at top level
    instance = base(2, "primary", base(1, "nested", base(0, "nested")))
    new_instance = serialiser.loads(serialiser.dumps(instance))
    assert new_instance.property_c.property_c.property_b == "nested"


def test_sensor_serialisation(serialiser):
    sensor = serialiser.loads(serialiser.dumps(TestSensor()))
    assert sensor.position is None
    assert sensor.orientation is None

    pos = StateVector([0, 1, 2])
    orientation = StateVector([0, np.pi/2, np.pi/4])
    sensor = serialiser.loads(serialiser.dumps(
        TestSensor(position=pos, orientation=orientation)))
    assert np.allclose(sensor.position, pos)
    assert np.allclose(sensor.orientation, orientation)


def test_particle_state(serialiser):
    state = ParticleState(
        StateVectors([[1, 2, 3], [4, 5, 6]]), weight=np.array([0.2, 0.3, 0.5]))
    data = serialiser.dumps(state)
    assert b'log_weight' in data
    assert b'weight' not in data.replace(b'log_weight', b'')
    new_state = serialiser.loads(data)
    assert np.allclose(new_state.state_vector, state.state_vector)
    assert np.allclose(new_state.log_weight, state.log_weight)
    assert np.allclose(new_state.weight.astype(float), state.weight.astype(float))


class _Custom:
    def __init__(self, value):
        self.value = value


def test_register(serialiser):
    serialiser.register(
        _Custom, lambda encoder, obj: encode_object(encoder, 'custom.Custom', obj.value),
        name='custom.Custom')
    new_instance = serialiser.loads(serialiser.dumps([_Custom(5)]))[0]
    assert isinstance(new_instance, _Custom)
    assert new_instance.value == 5


def test_register_subclasses(base, serialiser):
    class _SubBase(base):
        pass

    serialiser.register(
        base, lambda encoder, obj: encode_object(encoder, 'custom.Base', obj.property_b),
        decoder=lambda value: f"decoded {value}", name='custom.Base', subclasses=True)
    assert serialiser.loads(serialiser.dumps([base(1, "a"), _SubBase(2, "b")])) == \
        ["decoded a", "decoded b"]


def test_dump_load_path(tmpdir, serialiser):
    data = [1, 2, 3]
    path = Path(tmpdir.join('dump_file.cbor'))
    serialiser.dump(data, path)
    assert serialiser.load(path) == data
    with path.open('rb') as file:
        assert serialiser.load(file) == data


def test_dump_all(tmpdir, serialiser, base):
    instance = base(2, "20")
    documents = [[i, i + 1, i + 2, instance, instance] for i in range(5)]
    path = Path(tmpdir.join('dump_file.cbor'))
    serialiser.dump_all(documents, path)

    read_documents = list(serialiser.load_all(path))
    assert len(read_documents) == len(documents)
    for read_document, document in zip(read_documents, documents):
        assert read_document[:3] == document[:3]
        assert read_document[3].property_b == "20"
        if serialiser.value_sharing:
            assert read_document[3] is read_document[4]
    # References are within a document only
    assert read_documents[0][3] is not read_documents[1][3]

    read_documents = list(serialiser.load_all(path.read_bytes()))
    assert [document[:3] for document in read_documents] == \
        [document[:3] for document in documents]


def test_iter_json_compatible(base):
    from ..cbor import iter_json_compatible

    instance = base(2, "20")
    data = CBOR().dumps({
        'time': datetime.datetime(2024, 1, 2, 3, 4, 5),
        'component': instance,
        'shared': instance,
        'array': StateVector([1., 2.]),
        'int_array': np.array([[1, 2], [3, 4]], dtype=np.int32),
        'bearing': Bearing(0.5),
        'timedelta': datetime.timedelta(seconds=1.5),
        'set': {1},
        'tuple_key': {(1, 2): 'a'},
        'unknown_type': cbor2.CBORTag(27, ['not.a.real.Type', {'a': 1}]),
        'unknown_tag': cbor2.CBORTag(12345, 'value'),
        'bytes': b'\x01\x02',
    })

    items = list(iter_json_compatible(data))
    assert len(items) == 1
    item = items[0]
    json.dumps(item)  # Should be JSON serialisable
    component = {'!stonesoup.tests.conftest._TestBase': {'property_a': 2, 'property_b': '20'}}
    assert item == {
        'time': '2024-01-02T03:04:05',
        'component': component,
        'shared': component,  # Repeated in full
        'array': {'!stonesoup.types.array.StateVector': [[[1.], [2.]]]},
        'int_array': [[1, 2], [3, 4]],
        'bearing': {'!stonesoup.types.angle.Bearing': [0.5]},
        'timedelta': {'!datetime.timedelta': [0, 1, 500000]},
        'set': [1],
        'tuple_key': {'[1, 2]': 'a'},
        'unknown_type': {'!not.a.real.Type': {'a': 1}},  # Not imported
        'unknown_tag': {'CBORTag:12345': 'value'},
        'bytes': '0102',
    }


def test_iter_json_compatible_sequence(tmpdir):
    from ..cbor import iter_json_compatible

    path = Path(tmpdir.join('file.cbor'))
    CBOR().dump_all([{'a': i} for i in range(3)], path)
    assert list(iter_json_compatible(path)) == [{'a': 0}, {'a': 1}, {'a': 2}]
    with path.open('rb') as file:
        assert list(iter_json_compatible(file)) == [{'a': 0}, {'a': 1}, {'a': 2}]


@pytest.mark.parametrize('compact', [True, False], ids=['compact', 'indented'])
def test_main(tmpdir, capsys, base, compact):
    from ..cbor import main

    path = Path(tmpdir.join('file.cbor'))
    CBOR().dump_all([{'time': datetime.datetime(2024, 1, 1, 0, 0, i), 'component': base(i, "a")}
                     for i in range(2)], path)
    main(['--compact', str(path)] if compact else [str(path)])

    output = capsys.readouterr().out
    if compact:
        lines = output.splitlines()
        assert len(lines) == 2
        items = [json.loads(line) for line in lines]
    else:
        decoder = json.JSONDecoder()
        items, index = [], 0
        while index < len(output.strip()):
            item, index = decoder.raw_decode(output, index)
            items.append(item)
            while index < len(output) and output[index].isspace():
                index += 1
        assert '\n  ' in output  # Indented
    assert items == [
        {'time': f'2024-01-01T00:00:0{i}',
         'component': {'!stonesoup.tests.conftest._TestBase': {
             'property_a': i, 'property_b': 'a'}}}
        for i in range(2)]


def test_main_stdin(capsys, monkeypatch):
    import io
    import sys
    from ..cbor import main

    stdin = io.TextIOWrapper(io.BytesIO(CBOR().dumps([1, 2]) + CBOR().dumps({'a': 3})))
    monkeypatch.setattr(sys, 'stdin', stdin)
    main(['--compact', '-'])
    assert capsys.readouterr().out.splitlines() == ['[1, 2]', '{"a": 3}']


def test_main_module(tmpdir):
    import os
    import subprocess
    import sys

    import stonesoup

    path = Path(tmpdir.join('file.cbor'))
    CBOR().dump({'a': [1, 2]}, path)
    # Ensure same Stone Soup as under test is used
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [str(Path(stonesoup.__file__).parent.parent), env.get('PYTHONPATH', '')])
    result = subprocess.run(
        [sys.executable, '-W', 'error::RuntimeWarning', '-m', 'stonesoup.serialise.cbor',
         '--compact', str(path)],
        capture_output=True, text=True, env=env, check=True)
    assert result.stdout == '{"a": [1, 2]}\n'
