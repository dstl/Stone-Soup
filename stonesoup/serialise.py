# -*- coding: utf-8 -*-
"""Provides an ability to serialise Stone Soup objects into and from YAML.

Stone Soup utilises YAML_ for serialisation. The :doc:`stonesoup.base`
feature of components is exploited in order to store the data of the
components and data types.

.. _YAML: http://yaml.org/"""
import datetime
import warnings
from io import StringIO
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from importlib import import_module

import numpy as np
import ruamel.yaml
from ruamel.yaml.constructor import ConstructorError

from .base import Base


class YAML:
    """Class for YAML serialisation."""
    tag_prefix = '!{}.'.format(__name__.split('.', 1)[0])

    def __init__(self):
        self._yaml = ruamel.yaml.YAML()
        self._yaml.default_flow_style = False

        # NumPy
        self._yaml.representer.add_multi_representer(
            np.ndarray, self.ndarray_to_yaml)
        self._yaml.constructor.add_constructor(
            "!numpy.ndarray", self.ndarray_from_yaml)

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
        return representer.represent_omap(
            cls.yaml_tag(type(node)),
            OrderedDict((name, getattr(node, name))
                        for name, property_ in type(node).properties.items()
                        if getattr(node, name) is not property_.default))

    @classmethod
    def declarative_from_yaml(cls, constructor, tag_suffix, node):
        """Convert YAML to declarative class instances."""
        try:
            class_ = cls._get_class(tag_suffix)
        except ModuleNotFoundError:
            raise ConstructorError(
                "while constructing a Stone Soup component", node.start_mark,
                "unable to find component {!r}".format(tag_suffix),
                node.start_mark)
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
            raise ModuleNotFoundError("Unable to find {!r}".format(tag))
        return classes[0]

    def ndarray_to_yaml(self, representer, node):
        """Convert numpy.ndarray to YAML."""
        if node.ndim > 1:
            array = [self._yaml.seq(row) for row in node.tolist()]
            [seq.fa.set_flow_style() for seq in array]
        else:
            array = node.tolist()
        return representer.represent_sequence(
            "!numpy.ndarray", array)

    @staticmethod
    def ndarray_from_yaml(constructor, node):
        """Convert YAML to numpy.ndarray."""
        return np.array(constructor.construct_sequence(node, deep=True))

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
