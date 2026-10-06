"""Provides an ability to serialise Stone Soup objects into and from CBOR.

CBOR_ is a binary data format, which is typically much faster to read and write, and more compact,
than YAML (see :mod:`stonesoup.serialise.yaml`), at the expense of not being human readable. It is
implemented via the cbor2_ library.

Similar features to YAML are supported:

- Stone Soup components (and other types) are stored with their type name, so they can be
  dynamically loaded. This uses CBOR tag 27 (`serialised object with type name and constructor
  arguments`_), with value ``[type_name, *args]``. For :class:`~.Base` subclasses, the single
  argument is a map of the declared properties (skipping any which are the default value).
- Objects referenced multiple times are only stored once, using value sharing (tags 28 and 29),
  similar to YAML anchors and aliases.
- Repeated strings (e.g. type and property names) are only stored once, using string references
  (tags 25 and 256).
- NumPy arrays are stored as binary typed arrays (RFC 8746, tags 64-87) within multi-dimensional
  arrays (tag 40).

Datetimes are stored as RFC 3339 strings (tag 0). As Stone Soup typically uses naive datetimes,
these are by default stored without a timezone offset, which isn't strictly compliant with
RFC 8949, but are loaded back as naive datetimes. Alternatively, a timezone can be set (see
:class:`~.CBOR`), which is assumed for naive datetimes, which are then loaded as timezone aware.

Multiple objects can be written one after another to the same file (a CBOR sequence, RFC 8742),
analogous to multiple YAML documents.

CBOR files can be viewed as JSON from the command line, without loading any Stone Soup
components (see :func:`iter_json_compatible`)::

    python -m stonesoup.serialise.cbor [--compact] FILE [FILE ...]

It is also possible to extend the serialisation for other types with Stone Soup, via
`stonesoup.serialise.cbor` entry point, typically expected to be used with
:mod:`stonesoup.plugins`. The entry point should point to a function which expects a single
argument, a :class:`~.CBOR` instance, on which :meth:`~.CBOR.register` can be called.

.. _CBOR: https://cbor.io/
.. _cbor2: https://cbor2.readthedocs.io/
.. _serialised object with type name and constructor arguments:
    http://cbor.schmorp.de/generic-object
"""
import argparse
import datetime
import json
import sys
import warnings
from collections import deque
from collections.abc import Mapping
from importlib.metadata import entry_points
from io import BytesIO
from os import PathLike
from pathlib import Path

import cbor2
import numpy as np

from ..base import Base, Property
from ..sensor.sensor import Sensor
from ..types.angle import Angle
from ..types.array import Matrix, StateVector
from ..types.numeric import Probability
from ..types.state import ParticleState
from .yaml import get_class

__all__ = ['CBOR', 'encode_object', 'iter_json_compatible']

# CBOR major types
_MAJOR_ARRAY = 4
_MAJOR_MAP = 5
_MAJOR_TAG = 6

# CBOR tags
TAG_DATETIME_STRING = 0
TAG_OBJECT = 27
TAG_MULTI_DIM_ARRAY = 40
TAG_SET = 258


def _all_subclasses(class_):
    yield class_
    for subclass in class_.__subclasses__():
        yield from _all_subclasses(subclass)


def type_name(class_):
    """Return type name for class, as stored in CBOR.

    Constructed from module and class name (same as YAML tag, without the leading "!")."""
    return f"{class_.__module__}.{class_.__qualname__}"


def _typed_array_tag(dtype):
    """Return RFC 8746 typed array tag for NumPy dtype, or `None` if not supported."""
    if dtype.kind not in 'uif' or dtype.itemsize not in (1, 2, 4, 8):
        return None
    if dtype.kind == 'f':
        if dtype.itemsize == 1:
            return None
        size_bits = dtype.itemsize.bit_length() - 2  # 2->0, 4->1, 8->2
    else:
        size_bits = dtype.itemsize.bit_length() - 1  # 1->0, 2->1, 4->2, 8->3
    little_endian = dtype.byteorder == '<' or (dtype.byteorder == '=' and np.little_endian)
    if dtype.itemsize == 1:
        little_endian = False  # Single byte types have no endianness (bit used for clamped)
    return (0b01000000
            | (dtype.kind == 'f') << 4
            | (dtype.kind == 'i') << 3
            | little_endian << 2
            | size_bits)


def _typed_array_dtype(tag):
    """Return NumPy dtype for RFC 8746 typed array tag."""
    is_float = bool(tag & 0b10000)
    signed = bool(tag & 0b1000)
    little_endian = bool(tag & 0b100)
    size_bits = tag & 0b11
    if is_float:
        kind, itemsize = 'f', 2 << size_bits
    else:
        kind, itemsize = 'i' if signed else 'u', 1 << size_bits
        if itemsize == 1:
            little_endian = False  # Bit used for clamped
    return np.dtype(f"{'<' if little_endian else '>'}{kind}{itemsize}")


# Excluding reserved (76) and float128 (83, 87, not portable)
_TYPED_ARRAY_TAGS = {
    tag: _typed_array_dtype(tag) for tag in range(64, 88) if tag not in (76, 83, 87)}


class CBOR:
    """Class for CBOR serialisation in Stone Soup.

    Parameters
    ----------
    value_sharing : bool
        Store objects which are referenced multiple times only once, preserving their identity
        when loaded. Default `True`.
    string_referencing : bool
        Store repeated strings only once. Default `True`.
    timezone : datetime.tzinfo, optional
        Timezone assumed for naive datetimes, which are stored with the corresponding offset,
        and hence loaded as timezone aware datetimes. Default `None`, where naive datetimes are
        stored without a timezone offset (which isn't strictly compliant with RFC 8949), and
        loaded as naive datetimes. Timezone aware datetimes are unaffected.
    """

    def __init__(self, value_sharing=True, string_referencing=True, timezone=None):
        if timezone is not None and not isinstance(timezone, datetime.tzinfo):
            raise TypeError(
                f"timezone must be a datetime.tzinfo or None, not {type(timezone).__name__}")
        self.value_sharing = value_sharing
        self.string_referencing = string_referencing
        self.timezone = timezone
        self._all_encoders_cache = None
        self._registered_multi_encoders = []
        # Type specific encoders, by exact type (checked before `default`)
        self._encoders = {
            datetime.datetime: self._encode_datetime,
            datetime.timedelta: _encode_timedelta,
            deque: _encode_deque,
            Probability: _encode_probability,
        }
        # Type specific encoders, for subclasses (checked in order)
        self._multi_encoders = [
            (Base, _encode_declarative),
            (Angle, _encode_angle),
            (np.ndarray, _encode_ndarray),
            (np.bool_, lambda encoder, obj: encoder.encode(bool(obj))),
            (np.integer, lambda encoder, obj: encoder.encode(int(obj))),
            (np.floating, lambda encoder, obj: encoder.encode(float(obj))),
            (Path, _encode_path),
            (datetime.datetime, self._encode_datetime),
        ]
        # Decoders by type name, for objects stored with tag 27
        self._decoders = {
            type_name(datetime.timedelta): datetime.timedelta,
            type_name(deque): deque,
            type_name(Path): Path,
            type_name(Probability): lambda log_value: Probability(log_value, log_value=True),
        }
        for class_ in _all_subclasses(Angle):
            self._decoders[type_name(class_)] = class_
        for class_ in _all_subclasses(Matrix):
            self._decoders[type_name(class_)] = class_

        # Load additional custom serialisation
        for entry_point in entry_points(group="stonesoup.serialise.cbor"):
            try:
                entry_point.load()(self)
            except (ImportError, ModuleNotFoundError) as e:
                warnings.warn(f'Failed to load module. {e}')

        self._semantic_decoders = {
            TAG_OBJECT: self._decode_object,
            TAG_MULTI_DIM_ARRAY: _decode_multi_dim_array,
            # Avoid set contents being decoded as immutable (e.g. Track states as tuple)
            TAG_SET: lambda value, immutable: frozenset(value) if immutable else set(value),
        }
        for tag, dtype in _TYPED_ARRAY_TAGS.items():
            self._semantic_decoders[tag] = (
                lambda value, immutable, dtype=dtype:
                    np.frombuffer(value, dtype).astype(dtype.newbyteorder('=')))

    def register(self, class_, encoder=None, decoder=None, name=None, subclasses=False):
        """Register custom serialisation for a type.

        Parameters
        ----------
        class_ : type
            Type to register.
        encoder : callable, optional
            Function taking the :class:`cbor2.CBOREncoder` and object, which should encode the
            object. Typically this is done with :func:`encode_object`.
        decoder : callable, optional
            Function taking the arguments that were passed to :func:`encode_object`, and
            returning the decoded object. Default is to call `class_` with the arguments.
        name : str, optional
            Type name used in the serialised data. Default is module and qualified class name.
        subclasses : bool
            Whether `encoder` should also be used for subclasses of `class_`. Default `False`.
        """
        if encoder is not None:
            self._all_encoders_cache = None
            if subclasses:
                self._registered_multi_encoders.insert(0, (class_, encoder))
                self._multi_encoders.insert(0, (class_, encoder))
            else:
                self._encoders[class_] = encoder
        self._decoders[name or type_name(class_)] = decoder or class_

    def _encode_datetime(self, encoder, obj):
        """Encode datetime as RFC 3339 string (tag 0).

        Naive datetimes are assumed to be in :attr:`timezone` if set, otherwise they are stored
        without a timezone offset."""
        if obj.utcoffset() is None and self.timezone is not None:
            obj = obj.replace(tzinfo=self.timezone)
        encoder.encode_length(_MAJOR_TAG, TAG_DATETIME_STRING)
        encoder.encode(obj.isoformat())

    def _default(self, encoder, obj):
        for class_, class_encoder in self._multi_encoders:
            if isinstance(obj, class_):
                return class_encoder(encoder, obj)
        raise cbor2.CBOREncodeTypeError(f"cannot serialise type {type(obj)!r}")

    @property
    def _all_encoders(self):
        # Stone Soup components must be registered by exact type, as otherwise cbor2 will
        # encode those which are sequences or mappings (e.g. Track) natively, before trying
        # `default`. This also avoids `default` checking against each multi encoder.
        n_subclasses = len(Base._subclasses)
        if self._all_encoders_cache is None or self._all_encoders_cache[0] != n_subclasses:
            encoders = {class_: _encode_declarative for class_ in Base.subclasses}
            # Common types, to avoid `default` checking multi encoders
            encoders.update({class_: _encode_ndarray for class_ in _all_subclasses(np.ndarray)
                             if class_ is np.ndarray or issubclass(class_, Matrix)})
            encoders.update({class_: _encode_angle for class_ in _all_subclasses(Angle)})
            for registered_class, encoder in reversed(self._registered_multi_encoders):
                encoders.update(
                    {class_: encoder for class_ in _all_subclasses(registered_class)})
            encoders.update(self._encoders)
            self._all_encoders_cache = n_subclasses, encoders
        return self._all_encoders_cache[1]

    def _encoder(self, stream, data):
        return cbor2.CBOREncoder(
            stream,
            value_sharing=self.value_sharing,
            # Work around cbor2 only starting string reference namespace at first container,
            # which results in invalid references when the top level item isn't a container.
            string_referencing=(
                self.string_referencing and type(data) in (dict, list, tuple, set, frozenset)),
            encoders=self._all_encoders,
            default=self._default)

    def _decoder(self, stream):
        return cbor2.CBORDecoder(stream, semantic_decoders=self._semantic_decoders)

    def _decode_object(self, value, immutable):
        name, *args = value
        try:
            decoder = self._decoders[name]
        except KeyError:
            decoder = self._decoders[name] = _get_declarative_class(name)
        if isinstance(decoder, type) and issubclass(decoder, Base):
            try:
                return decoder(**args[0])
            except Exception as e:
                raise cbor2.CBORDecodeError(
                    f"while constructing Stone Soup component {name!r}: {e}") from e
        return decoder(*args)

    def dump(self, data, stream):
        """Serialise data to a (binary) stream or path.

        Multiple calls can be made with the same stream, to write a sequence of objects, which
        can be read with :meth:`load_all`."""
        if isinstance(stream, (str, PathLike)):
            with open(stream, 'wb') as file:
                return self.dump(data, file)
        self._encoder(stream, data).encode(data)

    def dumps(self, data):
        """Return serialised data as bytes."""
        stream = BytesIO()
        self.dump(data, stream)
        return stream.getvalue()

    def dump_all(self, documents, stream):
        """Serialise each of the documents to a (binary) stream or path, as a sequence."""
        if isinstance(stream, (str, PathLike)):
            with open(stream, 'wb') as file:
                return self.dump_all(documents, file)
        for document in documents:
            self.dump(document, stream)

    def load(self, stream):
        """Load single object from bytes, (binary) stream or path."""
        if isinstance(stream, (bytes, bytearray, memoryview)):
            stream = BytesIO(stream)
        elif isinstance(stream, (str, PathLike)):
            with open(stream, 'rb') as file:
                return self.load(file)
        return self._decoder(stream).decode()

    def loads(self, data):
        """Load single object from bytes."""
        return self.load(data)

    def load_all(self, stream):
        """Generator, loading each object in a sequence, from bytes, (binary) stream or path."""
        yield from _decode_all(stream, self._semantic_decoders)


def _decode_all(stream, semantic_decoders):
    """Generator, decoding each item in a CBOR sequence, from bytes, (binary) stream or path."""
    if isinstance(stream, (bytes, bytearray, memoryview)):
        stream = BytesIO(stream)
    elif isinstance(stream, (str, PathLike)):
        with open(stream, 'rb') as file:
            yield from _decode_all(file, semantic_decoders)
        return
    decoder = cbor2.CBORDecoder(stream, semantic_decoders=semantic_decoders)
    while True:
        try:
            yield decoder.decode()
        except cbor2.CBORDecodeEOF:
            return


def _get_declarative_class(name):
    # Only import modules within Stone Soup (including plugins); other components must
    # already be imported.
    if not name.startswith('stonesoup.') \
            and not any(type_name(class_) == name for class_ in Base.subclasses):
        raise cbor2.CBORDecodeError(f"unable to import component {name!r}: unknown type")
    try:
        class_ = get_class(f'!{name}')
    except (ImportError, AttributeError) as e:
        raise cbor2.CBORDecodeError(f"unable to import component {name!r}: {e}") from e
    if not (isinstance(class_, type) and issubclass(class_, Base)):
        raise cbor2.CBORDecodeError(f"unable to import component {name!r}: not a component")
    return class_


def encode_object(encoder, name, *args):
    """Encode object with type name and arguments (CBOR tag 27).

    Avoids creating temporary containers, which would otherwise be marked as shareable when
    value sharing is enabled."""
    encoder.encode_length(_MAJOR_TAG, TAG_OBJECT)
    encoder.encode_length(_MAJOR_ARRAY, len(args) + 1)
    encoder.encode(name)
    for arg in args:
        encoder.encode(arg)


@cbor2.shareable_encoder
def _encode_declarative(encoder, obj):
    """Encode declarative class instances.

    Stored as map of declared properties, skipping any which are the default value."""
    properties = type(obj).properties
    # Special case of a sensor with a default platform
    if isinstance(obj, Sensor) and obj._has_internal_controller:
        properties = dict(properties)
        properties['position'] = Property(StateVector)
        properties['orientation'] = Property(StateVector)
    # Special case of particle state, where weight is derived from log weight
    if isinstance(obj, ParticleState):
        properties = {name: property_ for name, property_ in properties.items()
                      if name != 'weight'}
    values = []
    for name, property_ in properties.items():
        value = getattr(obj, name)
        if value is not property_.default:
            values.append((name, value))

    encoder.encode_length(_MAJOR_TAG, TAG_OBJECT)
    encoder.encode_length(_MAJOR_ARRAY, 2)
    encoder.encode(type_name(type(obj)))
    encoder.encode_length(_MAJOR_MAP, len(values))
    for name, value in values:
        encoder.encode(name)
        encoder.encode(value)


def _encode_multi_dim_array(encoder, array):
    """Encode NumPy array as RFC 8746 multi-dimensional array (row-major)."""
    encoder.encode_length(_MAJOR_TAG, TAG_MULTI_DIM_ARRAY)
    encoder.encode_length(_MAJOR_ARRAY, 2)
    encoder.encode_length(_MAJOR_ARRAY, array.ndim)
    for dim in array.shape:
        encoder.encode(dim)
    tag = _typed_array_tag(array.dtype)
    if tag is not None:
        encoder.encode_length(_MAJOR_TAG, tag)
        encoder.encode(np.ascontiguousarray(array).tobytes())
    else:  # Fallback to generic array of items
        items = array.ravel().tolist() if array.dtype != object else list(array.ravel())
        encoder.encode_length(_MAJOR_ARRAY, len(items))
        for item in items:
            encoder.encode(item)


def _decode_multi_dim_array(value, immutable):
    dims, elements = value
    if not isinstance(elements, np.ndarray):
        elements = np.array(elements)
    return elements.reshape(dims)


@cbor2.shareable_encoder
def _encode_ndarray(encoder, obj):
    if type(obj) is np.ndarray:
        _encode_multi_dim_array(encoder, obj)
    else:  # Subclass e.g. StateVector
        encoder.encode_length(_MAJOR_TAG, TAG_OBJECT)
        encoder.encode_length(_MAJOR_ARRAY, 2)
        encoder.encode(type_name(type(obj)))
        _encode_multi_dim_array(encoder, obj.view(np.ndarray))


def _encode_angle(encoder, obj):
    encode_object(encoder, type_name(type(obj)), float(obj))


def _encode_probability(encoder, obj):
    encode_object(encoder, type_name(type(obj)), obj.log_value)


def _encode_timedelta(encoder, obj):
    encode_object(encoder, type_name(type(obj)), obj.days, obj.seconds, obj.microseconds)


def _encode_path(encoder, obj):
    encode_object(encoder, type_name(Path), str(obj))


@cbor2.shareable_encoder
def _encode_deque(encoder, obj):
    encode_object(encoder, type_name(deque), list(obj), obj.maxlen)


def _json_object(value, immutable):
    name, *args = value
    if len(args) == 1 and isinstance(args[0], Mapping):
        args = args[0]  # Component properties
    return {f"!{name}": args}


_JSON_SEMANTIC_DECODERS = {
    TAG_OBJECT: _json_object,
    TAG_MULTI_DIM_ARRAY: _decode_multi_dim_array,
    TAG_SET: lambda value, immutable: list(value),
    **{tag: lambda value, immutable, dtype=dtype: np.frombuffer(value, dtype)
       for tag, dtype in _TYPED_ARRAY_TAGS.items()},
}


def _to_json_compatible(obj):
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    elif isinstance(obj, Mapping):
        return {_to_json_key(key): _to_json_compatible(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple, set, frozenset)):
        return [_to_json_compatible(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return _to_json_compatible(obj.tolist())
    elif isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    elif isinstance(obj, (bytes, bytearray)):
        return obj.hex()
    elif isinstance(obj, cbor2.CBORTag):
        return {f"CBORTag:{obj.tag}": _to_json_compatible(obj.value)}
    else:
        return str(obj)


def _to_json_key(key):
    if key is None or isinstance(key, (str, bool, int, float)):
        return key
    return json.dumps(_to_json_compatible(key))


def iter_json_compatible(stream):
    """Generator of each item in a CBOR sequence, converted to JSON compatible types.

    This is intended for inspecting serialised data. Unlike :meth:`CBOR.load_all`, objects are
    not constructed, so no classes are imported. Instead, objects stored with their type name
    are represented as ``{"!type_name": properties}`` (or a list of arguments for non-component
    types), similar to YAML tags; arrays as nested lists; sets as lists; and datetimes as ISO
    8601 strings. Objects referenced multiple times are repeated in full.

    Parameters
    ----------
    stream : bytes, file or path
        CBOR data, (binary) stream or path.
    """
    for item in _decode_all(stream, _JSON_SEMANTIC_DECODERS):
        yield _to_json_compatible(item)


def main(args=None):
    """Command line interface, printing each item in CBOR file(s) as JSON."""
    parser = argparse.ArgumentParser(
        prog='python -m stonesoup.serialise.cbor',
        description="Print Stone Soup CBOR data as JSON, without loading any components.")
    parser.add_argument(
        'files', nargs='+', metavar='FILE', help="CBOR file(s) to print, or - for stdin")
    parser.add_argument(
        '--compact', action='store_true',
        help="print each item on a single line (JSON Lines), instead of indented")
    options = parser.parse_args(args)

    for file in options.files:
        stream = sys.stdin.buffer if file == '-' else file
        for item in iter_json_compatible(stream):
            print(json.dumps(item, indent=None if options.compact else 2))


if __name__ == '__main__':
    main()
