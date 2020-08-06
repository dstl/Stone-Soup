# -*- coding: utf-8 -*-
"""Provides an ability to serialise Stone Soup objects into and from YAML.

Stone Soup utilises YAML_ for serialisation. The :doc:`stonesoup.base`
feature of components is exploited in order to store the data of the
components and data types.

.. _YAML: http://yaml.org/"""
import datetime
import warnings
import sys
from io import StringIO
from collections import OrderedDict, deque
from functools import lru_cache
from pathlib import Path
from importlib import import_module

import numpy as np
import ruamel.yaml
from ruamel.yaml.constructor import ConstructorError

from .base import Base, Property
from .types.angle import Angle
from .types.array import Matrix, StateVector
from .types.numeric import Probability
from .sensor.sensor import Sensor

__all__ = ['YAML']

typ = 'stonesoup'


def init_typ(yaml):

    class _StoneSoupConstructor(yaml.Constructor):
        if sys.version_info < (3, 6):  # pragma: no cover
            from collections import OrderedDict
            yaml_multi_constructors = OrderedDict(yaml.Constructor.yaml_multi_constructors)

    class _StoneSoupRepresenter(yaml.Representer):
        if sys.version_info < (3, 6):  # pragma: no cover
            from collections import OrderedDict
            yaml_multi_representers = OrderedDict(yaml.Representer.yaml_multi_representers)

    yaml.Constructor = _StoneSoupConstructor
    yaml.Representer = _StoneSoupRepresenter


class YAML:
    """Class for YAML serialisation."""
    tag_prefix = '!{}.'.format(__name__.split('.', 1)[0])

    def __init__(self, typ='rt'):
        self._yaml = ruamel.yaml.YAML(typ=[typ, 'stonesoup'], plug_ins=['stonesoup.serialise'])
        self._yaml.default_flow_style = False

        # NumPy
        self._yaml.representer.add_multi_representer(
            np.ndarray, self.ndarray_to_yaml)
        self._yaml.constructor.add_constructor(
            "!numpy.ndarray", self.ndarray_from_yaml)
        self._yaml.representer.add_multi_representer(
            np.integer, self.numpy_int_to_yaml
        )
        self._yaml.representer.add_multi_representer(
            np.floating, self.numpy_float_to_yaml
        )

        # Datetime
        self._yaml.representer.add_representer(
            datetime.timedelta, self.timedelta_to_yaml)
        self._yaml.constructor.add_constructor(
            "!datetime.timedelta", self.timedelta_from_yaml)

        # Path
        self._yaml.representer.add_multi_representer(
            Path, self.path_to_yaml)
        self._yaml.constructor.add_constructor(
            "!pathlib.Path", self.path_from_yaml)

        # deque
        self._yaml.representer.add_representer(
            deque, self.deque_to_yaml)
        self._yaml.constructor.add_constructor(
            "!collections.deque", self.deque_from_yaml)
        # Probability
        self._yaml.representer.add_representer(
            Probability, self.probability_to_yaml)
        self._yaml.constructor.add_constructor(
            self.yaml_tag(Probability), self.probability_from_yaml)

        # Angle
        self._yaml.representer.add_multi_representer(
            Angle, self.angle_to_yaml)
        self._yaml.constructor.add_multi_constructor(
            '{}types.angle.'.format(self.tag_prefix), self.angle_from_yaml)

        # Array
        self._yaml.representer.add_multi_representer(
            Matrix, self.ndarray_to_yaml)
        self._yaml.constructor.add_multi_constructor(
            '{}types.array.'.format(self.tag_prefix), self.array_from_yaml)

        # Declarative classes
        self._yaml.representer.add_multi_representer(
            Base, self.declarative_to_yaml)
        self._yaml.constructor.add_multi_constructor(
            self.tag_prefix, self.declarative_from_yaml)

    def dump(self, data, stream, **kwargs):
        return self._yaml.dump(data, stream, **kwargs)

    def dumps(self, data, *args, **kwargs):
        """Return as a string."""
        stream = StringIO()
        self.dump(data, stream, *args, **kwargs)
        return stream.getvalue()

    def dump_all(self, documents, stream, **kwargs):
        return self._yaml.dump_all(self, documents, stream, **kwargs)

    def load(self, stream):
        return self._yaml.load(stream)

    def load_all(self, stream):
        yield from self._yaml.load_all(stream)

    @classmethod
    def yaml_tag(cls, class_):
        """Return YAML tag for object.

        Constructed from module and class name."""
        return "!{}.{}".format(class_.__module__, class_.__qualname__)

    @classmethod
    def declarative_to_yaml(cls, representer, node):
        """Convert declarative class instances to YAML.

        Store as mapping of declared properties, skipping any which are the
        default value."""
        node_properties = OrderedDict(type(node).properties)
        # Special case of a sensor with a default platform
        if isinstance(node, Sensor) and node._has_internal_platform:
            node_properties['position'] = Property(StateVector)
            node_properties['orientation'] = Property(StateVector)
        return representer.represent_omap(
            cls.yaml_tag(type(node)),
            OrderedDict((name, getattr(node, name))
                        for name, property_ in node_properties.items()
                        if getattr(node, name) is not property_.default))

    @classmethod
    def declarative_from_yaml(cls, constructor, tag_suffix, node):
        """Convert YAML to declarative class instances."""
        try:
            class_ = cls._get_class(tag_suffix)
        except ImportError:
            raise ConstructorError(
                "while constructing a Stone Soup component", node.start_mark,
                "unable to import component {!r}".format(tag_suffix),
                node.start_mark)
        # Must have deep construct here to ensure mutable sub-objects are fully created.
        constructor.deep_construct = True
        properties = [
            data
            for data in constructor.construct_yaml_omap(node)][0]
        try:
            return class_(**properties)
        except Exception as e:
            raise ConstructorError("while constructing Stone Soup component",
                                   node.start_mark, str(e), node.start_mark)

    @classmethod
    @lru_cache(None)
    def _get_class(cls, tag_suffix):
        tag = cls.tag_prefix + tag_suffix
        classes = [
            subclass
            for subclass in Base.subclasses
            if cls.yaml_tag(subclass) == tag]
        if len(classes) > 1:
            warnings.warn(
                "Multiple possible classes found for YAML tag {!r}".format(
                    tag), UserWarning)
        elif not classes:
            module_name, class_name = tag_suffix.rsplit(".", 1)
            module = import_module("..{}".format(module_name), __name__)
            classes = [getattr(module, class_name, None)]
        if classes[0] is None:
            raise ImportError("Unable to find {!r}".format(tag))
        return classes[0]

    @classmethod
    def probability_to_yaml(cls, representer, node):
        return representer.represent_scalar(cls.yaml_tag(type(node)), str(node))

    @staticmethod
    def probability_from_yaml(constructor, node):
        string = constructor.construct_scalar(node)
        if string.startswith('exp('):
            return Probability(float(string[4:-1]), log_value=True)
        else:
            return Probability(float(string))

    @classmethod
    def angle_to_yaml(cls, representer, node):
        return representer.represent_scalar(cls.yaml_tag(type(node)), str(node))

    @classmethod
    def angle_from_yaml(cls, constructor, tag_suffix, node):
        class_ = cls._get_class('types.angle.{}'.format(tag_suffix))
        return class_(float(constructor.construct_scalar(node)))

    def ndarray_to_yaml(self, representer, node):
        """Convert numpy.ndarray to YAML."""

        # If using "round trip" type, change flow style to make more readable
        if node.ndim > 1 and 'rt' in self._yaml.typ:
            array = [self._yaml.seq(row) for row in node.tolist()]
            [seq.fa.set_flow_style() for seq in array]
        else:
            array = node.tolist()
        return representer.represent_sequence(self.yaml_tag(type(node)), array)

    @staticmethod
    def ndarray_from_yaml(constructor, node):
        """Convert YAML to numpy.ndarray."""
        return np.array(constructor.construct_sequence(node, deep=True))

    @classmethod
    def array_from_yaml(cls, constructor, tag_suffix, node):
        """Convert YAML to numpy.ndarray."""
        class_ = cls._get_class('types.array.{}'.format(tag_suffix))
        return class_(constructor.construct_sequence(node, deep=True))

    @staticmethod
    def numpy_int_to_yaml(representer, node):
        """Convert numpy ints to YAML"""
        return representer.represent_int(node)

    @staticmethod
    def numpy_float_to_yaml(representer, node):
        """Convert numpy floats to YAML"""
        return representer.represent_float(node)

    @staticmethod
    def timedelta_to_yaml(representer, node):
        """Convert datetime.timedelta to YAML.

        Value is total number of seconds."""
        return representer.represent_scalar(
            "!datetime.timedelta", str(node.total_seconds()))

    @staticmethod
    def timedelta_from_yaml(constructor, node):
        """Convert YAML to datetime.timedelta.

        Value should be total number of seconds."""
        return datetime.timedelta(
            seconds=float(constructor.construct_scalar(node)))

    @staticmethod
    def path_to_yaml(representer, node):
        """Convert path to YAML.

        Value is total number of seconds."""
        return representer.represent_scalar(
            "!pathlib.Path", str(node))

    @staticmethod
    def path_from_yaml(constructor, node):
        """Convert YAML to datetime.timedelta.

        Value should be total number of seconds."""
        return Path(constructor.construct_scalar(node))

    @staticmethod
    def deque_to_yaml(representer, node):
        """Convert collections.deque to YAML"""
        return representer.represent_sequence(
            "!collections.deque",
            (list(node), node.maxlen))

    @staticmethod
    def deque_from_yaml(constructor, node):
        """Convert YAML to collections.deque"""
        iterable, maxlen = constructor.construct_sequence(node, deep=True)
        return deque(iterable, maxlen)
