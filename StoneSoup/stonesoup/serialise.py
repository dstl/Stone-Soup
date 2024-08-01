"""Provides an ability to serialise Stone Soup objects into and from YAML.

Stone Soup utilises YAML_ for serialisation. The :doc:`stonesoup.base`
feature of components is exploited in order to store the data of the
components and data types.

This module functions as a plug-in for ruamel.yaml_, specified by
:code:`typ='stonesoup'`, but for convenience it is recommended to
use :class:`~.stonesoup.serialise.YAML` which defaults with the plug-in
enabled.

It is also possible to extend the serialisation for other types with
Stone Soup, via `stonesoup.serialise.yaml` entry point, typically
expected to be used with :mod:`stonesoup.plugins`. The entry point
should point to a function which expects a single argument, a
:class:`ruamel.yaml.YAML` instance.

For example:

.. code-block:: python

    setup(
        ...
        entry_points={
            'stonesoup.plugins': 'my_plugin = my_package',
            'stonesoup.serialise.yaml': 'my_plugin = my_package:yaml_init_func}
        ...
    )


.. _YAML: http://yaml.org/
.. _ruamel.yaml: https://yaml.readthedocs.io/
"""
import datetime
import warnings
from io import StringIO
from collections import OrderedDict, deque
from functools import lru_cache
from pathlib import Path
from importlib import import_module
from importlib.metadata import entry_points

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
    # Load additional custom serialisation
    eps = entry_points()
    try:
        entrypoints = eps['stonesoup.serialise.yaml']
    except KeyError:
        entrypoints = []
    for entry_point in entrypoints:
        try:
            entry_point.load()(yaml)
        except (ImportError, ModuleNotFoundError) as e:
            warnings.warn(f'Failed to load module. {e}')

    # NumPy
    yaml.representer.add_multi_representer(np.ndarray, ndarray_to_yaml)
    yaml.constructor.add_constructor("!numpy.ndarray", ndarray_from_yaml)
    yaml.representer.add_multi_representer(np.integer, yaml.representer.yaml_representers[int])
    yaml.representer.add_multi_representer(np.floating, npfloating_as_yaml)

    # Datetime
    yaml.representer.add_representer(datetime.timedelta, timedelta_to_yaml)
    yaml.constructor.add_constructor("!datetime.timedelta", timedelta_from_yaml)

    # Path
    yaml.representer.add_multi_representer(Path, path_to_yaml)
    yaml.constructor.add_constructor("!pathlib.Path", path_from_yaml)

    # deque
    yaml.representer.add_representer(deque, deque_to_yaml)
    yaml.constructor.add_constructor("!collections.deque", deque_from_yaml)
    # Probability
    yaml.representer.add_representer(Probability, probability_to_yaml)
    yaml.constructor.add_constructor(yaml_tag(Probability), probability_from_yaml)

    # Angle
    yaml.representer.add_multi_representer(Angle, angle_to_yaml)
    yaml.constructor.add_multi_constructor('!stonesoup.types.angle.', angle_from_yaml)

    # Array
    yaml.representer.add_multi_representer(Matrix, ndarray_to_yaml)
    yaml.constructor.add_multi_constructor('!stonesoup.types.array.', array_from_yaml)

    # Declarative classes
    yaml.representer.add_multi_representer(Base, declarative_to_yaml)
    yaml.constructor.add_multi_constructor('!stonesoup.', declarative_from_yaml)


class YAML(ruamel.yaml.YAML):
    """Class for YAML serialisation in Stone Soup."""

    def __init__(self, **kwargs):
        typ = kwargs.pop('typ', ['rt'])
        if isinstance(typ, str):
            typ = [typ]
        typ.append('stonesoup')
        if kwargs.get('plug_ins') is None:
            kwargs['plug_ins'] = []
        kwargs['plug_ins'].append('stonesoup.serialise')

        super().__init__(typ=typ, **kwargs)
        self.representer.default_flow_style = False
        self.representer.sort_base_mapping_type_on_output = False

    def dumps(self, data, *args, **kwargs):
        """Return as a string."""
        stream = StringIO()
        self.dump(data, stream, *args, **kwargs)
        return stream.getvalue()


def yaml_tag(class_):
    """Return YAML tag for object.

    Constructed from module and class name."""
    return f"!{class_.__module__}.{class_.__qualname__}"


def declarative_to_yaml(representer, node):
    """Convert declarative class instances to YAML.

    Store as mapping of declared properties, skipping any which are the
    default value."""
    node_properties = OrderedDict(type(node).properties)
    # Special case of a sensor with a default platform
    if isinstance(node, Sensor) and node._has_internal_controller:
        node_properties['position'] = Property(StateVector)
        node_properties['orientation'] = Property(StateVector)
    return representer.represent_omap(
        yaml_tag(type(node)),
        OrderedDict((name, getattr(node, name))
                    for name, property_ in node_properties.items()
                    if getattr(node, name) is not property_.default))


def declarative_from_yaml(constructor, tag_suffix, node):
    """Convert YAML to declarative class instances."""
    try:
        class_ = get_class(f'!stonesoup.{tag_suffix}')
    except ImportError:
        raise ConstructorError(
            "while constructing a Stone Soup component", node.start_mark,
            f"unable to import component 'stonesoup.{tag_suffix}'", node.start_mark)
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


@lru_cache(None)
def get_class(tag):
    classes = [
        subclass
        for subclass in Base.subclasses
        if yaml_tag(subclass) == tag]
    if len(classes) > 1:
        warnings.warn(
            f"Multiple possible classes found for YAML tag {tag!r}", UserWarning)
    elif not classes:
        module_name, class_name = tag.lstrip('!').rsplit(".", 1)
        module = import_module(module_name)
        classes = [getattr(module, class_name, None)]
    if classes[0] is None:
        raise ImportError(f"Unable to find {tag!r}")
    return classes[0]


def probability_to_yaml(representer, node):
    return representer.represent_scalar(yaml_tag(type(node)), str(node))


def probability_from_yaml(constructor, node):
    string = constructor.construct_scalar(node)
    if string.startswith('exp('):
        return Probability(float(string[4:-1]), log_value=True)
    else:
        return Probability(float(string))


def angle_to_yaml(representer, node):
    return representer.represent_scalar(yaml_tag(type(node)), str(node))


def angle_from_yaml(constructor, tag_suffix, node):
    class_ = get_class(f'!stonesoup.types.angle.{tag_suffix}')
    return class_(float(constructor.construct_scalar(node)))


def npfloating_as_yaml(representer, node):
    """Convert np.floating to YAML."""
    return representer.yaml_representers[float](representer, float(node))


def ndarray_to_yaml(representer, node):
    """Convert numpy.ndarray to YAML."""

    # If using "round trip" type, change flow style to make more readable
    if node.ndim > 1 and 'rt' in representer.dumper.typ:
        array = [representer.dumper.seq(row) for row in node.tolist()]
        [seq.fa.set_flow_style() for seq in array]
    else:
        array = node.tolist()
    return representer.represent_sequence(yaml_tag(type(node)), array)


def ndarray_from_yaml(constructor, node):
    """Convert YAML to numpy.ndarray."""
    return np.array(constructor.construct_sequence(node, deep=True))


def array_from_yaml(constructor, tag_suffix, node):
    """Convert YAML to numpy.ndarray."""
    class_ = get_class(f'!stonesoup.types.array.{tag_suffix}')
    return class_(constructor.construct_sequence(node, deep=True))


def timedelta_to_yaml(representer, node):
    """Convert datetime.timedelta to YAML.

    Value is total number of seconds."""
    return representer.represent_scalar("!datetime.timedelta", str(node.total_seconds()))


def timedelta_from_yaml(constructor, node):
    """Convert YAML to datetime.timedelta.

    Value should be total number of seconds."""
    return datetime.timedelta(seconds=float(constructor.construct_scalar(node)))


def path_to_yaml(representer, node):
    """Convert path to YAML.

    Value is total number of seconds."""
    return representer.represent_scalar("!pathlib.Path", str(node))


def path_from_yaml(constructor, node):
    """Convert YAML to datetime.timedelta.

    Value should be total number of seconds."""
    return Path(constructor.construct_scalar(node))


def deque_to_yaml(representer, node):
    """Convert collections.deque to YAML"""
    return representer.represent_sequence("!collections.deque", (list(node), node.maxlen))


def deque_from_yaml(constructor, node):
    """Convert YAML to collections.deque"""
    iterable, maxlen = constructor.construct_sequence(node, deep=True)
    return deque(iterable, maxlen)
